"""UCB1算法模块：UCB1算法代理类

公开接口：
- UCB1Agent: UCB1算法代理类
"""

from __future__ import annotations

from typing import cast, Callable

from core.agent import BaseAgent
from core.environment import RLEnv
from core.schemas import Metrics
from .schemas import UCB1RewardsState


class UCBAgent(BaseAgent):
    """UCB1算法代理类"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        ucb1_algorithm: Callable[..., int],
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """UCB1算法代理初始化

        Args:
            name (str): 代理名称
            env (RLEnv): 环境
            ucb1_algorithm (Callable[..., int]): UCB1算法
            convergence_threshold (float, optional): 收敛阈值
            convergence_min_steps (int, optional): 最小收敛步数
            seed (int, optional): 随机种子
        """
        super().__init__(
            name=name, 
            env=env, 
            convergence_threshold=convergence_threshold,
            convergence_min_steps=convergence_min_steps,
            seed=seed
        )

        self.rewards = UCB1RewardsState.from_env(env)
        self.ucb1_algorithm = ucb1_algorithm

    def act(self, **kwargs) -> int:
        """选择行动（拉动哪个老虎机）"""
        choice = self.ucb1_algorithm(cast(UCB1RewardsState, self.rewards), self.rng, self.steps)
        self.steps += 1
        return choice

    def pull_machine(self, machine_id: int) -> int:
        """拉动指定机器并更新状态"""
        reward = self.env.pull(machine_id)
        self._update_q_value(machine_id, reward)
        self._check_convergence()
        return reward

    def _update_q_value(self, machine_id: int, reward: int):
        """使用增量方式更新 Q 值"""
        ucb_rewards = cast(UCB1RewardsState, self.rewards)
        ucb_rewards.counts[machine_id] += 1
        count = ucb_rewards.counts[machine_id]
        ucb_rewards.values[machine_id] += reward

        # Q(A) ← Q(A) + (R - Q(A)) / N(A)
        old_q = ucb_rewards.q_values[machine_id]
        ucb_rewards.q_values[machine_id] = old_q + (reward - old_q) / count

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
