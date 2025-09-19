"""UCB1算法模块：UCB1算法代理类

公开接口：
- UCBAgent: UCB1算法代理类
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from core import BaseAgent, BaseAlgorithm
from core.environment import RLEnv
from .schemas import UCB1RewardsState, UCB1AlgorithmType


class UCBAgent(BaseAgent[UCB1RewardsState, "UCB1Algorithm"]):
    """UCB1算法代理类"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        algorithm: "UCB1Algorithm",
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        constant_stepsize: float = 0.01,
        seed: int = 42,
    ) -> None:
        """UCB1算法代理初始化

        Args:
            name (str): 名称
            env (RLEnv): 环境
            algorithm (UCB1Algorithm): 所使用的UCB1算法
            convergence_threshold (float, optional): 达到收敛条件的阈值
            convergence_min_steps (int, optional): 达到收敛条件的最小步数
            constant_stepsize (float, optional): 常数步长
            seed (int, optional): 种子
        """
        super().__init__(
            name=name,
            env=env,
            algorithm=algorithm,
            convergence_threshold=convergence_threshold,
            convergence_min_steps=convergence_min_steps,
            seed=seed,
        )

        self.rewards = UCB1RewardsState.from_env(env=env)
        self.constant_stepsize = constant_stepsize

    def act(self, **_) -> int:
        """选择行动（拉动哪个老虎机）"""
        choice = self.algorithm.run()

        self.steps += 1
        return choice

    def pull_machine(self, machine_id: int) -> int:
        """拉动指定机器并更新状态"""
        reward = self.env.pull(machine_id, self.steps)
        self._update_q_value(machine_id, reward)
        self._check_convergence()
        return reward

    def _update_q_value(self, machine_id: int, reward: int):
        """使用增量方式更新 Q 值"""
        self.rewards.counts[machine_id] += 1
        count = self.rewards.counts[machine_id]
        self.rewards.values[machine_id] += reward

        old_q = self.rewards.q_values[machine_id]
        if self.constant_stepsize:
            self.rewards.q_values[machine_id] = old_q + self.constant_stepsize * (reward - old_q)
        else:
            self.rewards.q_values[machine_id] = old_q + (reward - old_q) / count


@dataclass
class UCBInitState:
    ucb_inited: bool = False
    ucb_inited_index: int = 0


class UCB1Algorithm(BaseAlgorithm[UCBAgent, UCB1AlgorithmType]):
    def __init__(self, ucb1_type: UCB1AlgorithmType = UCB1AlgorithmType.UCB1) -> None:
        super().__init__(ucb1_type, UCBAgent)
        self.ucb_init_state = UCBInitState()

    def set_agent(self, agent: UCBAgent) -> None:
        super().set_agent(agent)

    def run(self) -> int:
        return self.ucb1()

    def ucb1(self) -> int:
        """UCB1 算法"""
        rewards = self.agent.rewards
        steps = self.agent.steps

        for i in range(len(rewards.counts)):
            if rewards.counts[i] == 0:
                return i

        if not self.ucb_init_state.ucb_inited:
            self.ucb_init_state.ucb_inited = True

        log_steps = np.log(steps) if steps > 0 else 0.0
        counts_np = np.array(rewards.counts, dtype=np.float64)
        q_values = rewards.q_values

        ucb_values = q_values + np.sqrt(2 * log_steps / np.maximum(counts_np, 1))

        rewards.ucb_values = ucb_values

        return int(np.argmax(ucb_values))
