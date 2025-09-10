"""贪婪算法模块：贪婪算法代理类

公开接口：
- GreedyAgent: 贪婪算法代理类，继承自 BaseAgent
"""

from __future__ import annotations

from typing import Callable

from core.agent import BaseAgent
from core.environment import RLEnv
from core.schemas import RewardsState, Metrics
from .config import EpsilonDecreasingConfig, EpsilonDecreasingState


class GreedyAgent(BaseAgent):
    """贪婪算法代理类，继承自 BaseAgent"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        greedy_algorithm: Callable[..., int],
        epsilon_config: EpsilonDecreasingConfig = EpsilonDecreasingConfig(),
        optimistic_init: bool = False,
        optimistic_times: int = 1,
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """贪婪算法代理初始化

        Args:
            name (str): 名称
            env (RLEnv): 环境
            greedy_algorithm (Callable[..., int]): 所使用的贪婪算法
            epsilon_config (EpsilonDecreasingConfig, optional): 退火配置
            optimistic_init (bool, optional): 是否使用乐观初始化
            optimistic_times (int, optional): 乐观初始化的次数
            convergence_threshold (float, optional): 达到收敛条件的阈值（最佳臂选择率）
            convergence_min_steps (int, optional): 达到收敛条件的最小次数，至少要达到这个次数才能算作收敛
            seed (int, optional): 种子
        """
        super().__init__(name=name, env=env, seed=seed)

        self.greedy_algorithm = greedy_algorithm
        self.episode_state = EpsilonDecreasingState(
            epsilon=epsilon_config.start_epsilon,
            decay=epsilon_config.decay,
            min_epsilon=epsilon_config.min_epsilon,
        )
        self.rewards: RewardsState = RewardsState.from_env(
            env,
            optimistic_init=optimistic_init,
            optimistic_times=optimistic_times,
        )
        self.optimistic_init = optimistic_init
        self.convergence_threshold = convergence_threshold
        self.convergence_min_steps = convergence_min_steps
        self.optimistic_inited = False
        self.convergence_steps = 0

    def act(self, **kwargs) -> int:
        """选择行动（拉动哪个老虎机）"""
        if self.optimistic_init and not self.optimistic_inited:
            machine_id: int | None = None
            for r in self.rewards.q_values_optimistic:
                if r > 0:
                    machine_id = self.rewards.q_values_optimistic.index(r)
                    self.rewards.q_values_optimistic[machine_id] -= 1
                    return machine_id

            self.optimistic_inited = True
        return self.greedy_algorithm(self.rewards, self.rng, **kwargs)

    def pull_machine(self, machine_id: int) -> int:
        """拉动指定机器并更新状态"""
        reward = self.env.pull(machine_id)
        self._update_q_value(machine_id, reward)
        self._check_convergence()
        return reward

    def _update_q_value(self, machine_id: int, reward: int):
        """使用增量方式更新 Q 值"""
        self.rewards.counts[machine_id] += 1
        count = self.rewards.counts[machine_id]
        self.rewards.values[machine_id] += reward

        # Q(A) ← Q(A) + (R - Q(A)) / N(A)
        old_q = self.rewards.q_values[machine_id]
        self.rewards.q_values[machine_id] = old_q + (reward - old_q) / count

    def _check_convergence(self):
        """检查是否达到收敛条件"""
        if self.steps < self.convergence_min_steps or self.convergence_steps > 0:
            return

        if self.optimal_rate() >= self.convergence_threshold:
            self.convergence_steps = self.steps
            print(f"达到收敛时的步数: {self.convergence_steps}")
            return

    def regret(self) -> float:
        """计算后悔值"""
        return self.env.best_reward(self.steps) - sum(self.rewards.values)

    def regret_rate(self) -> float:
        """计算后悔率"""
        if self.steps == 0:
            return 0.0
        return self.regret() / self.env.best_reward(self.steps)

    def optimal_rate(self) -> float:
        """计算最优臂选择率"""
        if self.steps == 0:
            return 0.0
        return self.rewards.counts[-1] / self.steps

    def metric(self) -> Metrics:
        """获取当前指标"""
        return Metrics(
            regret=self.regret(),
            regret_rate=self.regret_rate(),
            rewards=self.rewards.model_copy(deep=True),
            optimal_rate=self.optimal_rate(),
        )
