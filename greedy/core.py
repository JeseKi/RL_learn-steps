from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import random
import numpy as np

from pydantic import BaseModel, Field


class SlotMachine:
    """老虎机，每次拉动有一定概率获得奖励"""

    def __init__(self, reward_probability: float, seed: int = 42) -> None:
        self.reward_probability = reward_probability
        self.rng = random.Random(seed)

    def pull(self) -> int:
        return 1 if self.rng.random() < self.reward_probability else 0


class RLEnv:
    """环境中包含多个老虎机，默认为 10 个"""

    def __init__(self, machine_count: int = 10, seed: int = 42) -> None:
        self.machines: List[SlotMachine] = []

        for i in range(machine_count):
            self.machines.append(
                SlotMachine(reward_probability=(i + 1) / (machine_count + 1), seed=seed)
            )

        self.best_reward_machine: SlotMachine = self.machines[
            -1
        ]  # 初始化代码中，对老虎机列表的初始化就是最后一个老虎机是奖励期望最高的

    def pull(self, machine_id: int) -> int:
        assert 0 <= machine_id < len(self.machines)
        return self.machines[machine_id].pull()

    def best_reward(self, steps: int) -> float:
        return self.machines[-1].reward_probability * steps


@dataclass
class EpsilonDecreasingState:
    epsilon: float
    decay: float
    min_epsilon: float


@dataclass
class EpsilonDecreasingConfig:
    start_epsilon: float = 1.0
    decay: float = 0.995
    min_epsilon: float = 0.01


class Rewards(BaseModel):
    """贪婪 Agent 获得的奖励记录"""

    values: List[float] = Field(description="每个机器的累积奖励")
    counts: List[float] = Field(description="每个机器的拉动次数")
    optimistic_init: bool = Field(
        default=False,
        description="是否使用乐观初始化：True表示初始Q值为1，False表示初始Q值为0",
    )
    optimistic_times: int = Field(
        default=1,
        description="乐观初始化的次数",
    )
    q_values: List[float] = Field(default_factory=list, description="每个机器的Q值")
    q_values_optimistic: List[float] = Field(default_factory=list, description="每个机器的乐观初始化Q值")

    @classmethod
    def from_env(
        cls,
        env: RLEnv,
        initial_value: float = 0.0,
        initial_count: float = 0.0,
        optimistic_init: bool = False,
        optimistic_times: int = 1,
    ) -> Rewards:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env (RLEnv): 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (float, optional): counts 列表中每个元素的初始值，默认为 0.0。

        Returns:
            Rewards: 一个初始化好的 Rewards 实例。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
            optimistic_init=optimistic_init,
            optimistic_times=optimistic_times,
            q_values=[initial_value] * num_machines,
            q_values_optimistic=[optimistic_times] * num_machines,
        )


class Metrics(BaseModel):
    regret: float = Field(..., description="后悔值")
    regret_rate: float = Field(..., description="后悔率")
    rewards: Rewards = Field(..., description="奖励")
    optimal_rate: float = Field(..., description="最佳臂命中率")


class GreedyAgent:
    """我们的 Agent 默认使用贪婪算法，来找到最优的老虎机"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        greedy_algorithm: Callable[..., int],
        epsilon_config: EpsilonDecreasingConfig = EpsilonDecreasingConfig(),
        optimistic_init: bool = False,
        optimistic_times: int = 1,
        seed: int = 42,
    ) -> None:
        """贪婪算法的 Agent

        Args:
            name (str): 名称
            env (RLEnv): 环境
            greedy_algorithm (Callable[..., int]): 所使用的贪婪算法
            epsilon_config (EpsilonDecreasingConfig, optional): 退火配置
            seed (int, optional): 种子
        """

        self.name = name
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.greedy_algorithm = greedy_algorithm
        self.episode_state = EpsilonDecreasingState(
            epsilon=epsilon_config.start_epsilon,
            decay=epsilon_config.decay,
            min_epsilon=epsilon_config.min_epsilon,
        )
        self.env = env
        self.rewards: Rewards = Rewards.from_env(
            env,
            optimistic_init=optimistic_init,
            optimistic_times=optimistic_times,
        )
        self.optimistic_init = optimistic_init
        
        self.steps: int = 0
        self.metrics_history: List[Tuple[Rewards, Metrics, int]] = []
        self.optimistic_inited = False

    def act(self, **kwargs) -> int:
        """选择拉动哪个老虎机，传入一个指定的贪婪算法，根据当前的奖励情况，选择一个老虎机"""
        self.steps += 1
        if self.optimistic_init and not self.optimistic_inited:
            machine_id: int | None = None
            for r in self.rewards.q_values_optimistic:
                if r > 0:
                    machine_id = self.rewards.q_values_optimistic.index(r)
                    self.rewards.q_values_optimistic[machine_id] -= 1
                    return machine_id
            
            self.optimistic_inited = True
        return self.greedy_algorithm(self.rewards, self.rng, **kwargs)

    def regret(self):
        """计算后悔值"""
        regret = self.env.best_reward(self.steps) - sum(self.rewards.values)
        return regret

    def regret_rate(self):
        """计算后悔率"""
        return self.regret() / self.env.best_reward(self.steps)

    def optimal_rate(self):
        return self.rewards.counts[-1] / self.steps if self.steps else 0

    def metric(self):
        return Metrics(
            regret=self.regret(),
            regret_rate=self.regret_rate(),
            rewards=self.rewards.model_copy(deep=True),
            optimal_rate=self.optimal_rate(),
        )

    def pull_machine(self, machine_id: int) -> int:
        reward = self.env.pull(machine_id)
        self._update_q_value(machine_id, reward)
        return reward
    
    def _update_q_value(self, machine_id: int, reward: int):
        """使用增量方式更新 Q 值"""
        self.rewards.counts[machine_id] += 1
        count = self.rewards.counts[machine_id]
        self.rewards.values[machine_id] += reward

        # Q(A) ← Q(A) + (R - Q(A)) / N(A)
        old_q = self.rewards.q_values[machine_id]
        self.rewards.q_values[machine_id] = old_q + (reward - old_q) / count