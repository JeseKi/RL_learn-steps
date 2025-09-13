"""UCB1算法模块：UCB1算法代理类

公开接口：
- UCBAgent: UCB1算法代理类
"""

from __future__ import annotations

from typing import cast, Callable
import numpy as np

from core.agent import BaseAgent
from core.environment import RLEnv
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
            seed=seed,
        )

        self.rewards = UCB1RewardsState.from_env(env=env)
        self.ucb1_algorithm = ucb1_algorithm

    def act(self, **_) -> int:
        """选择行动（拉动哪个老虎机）"""
        ucb_rewards = cast(UCB1RewardsState, self.rewards)
        steps = self.steps

        if not ucb_rewards.ucb_states.ucb_inited:
            choice = ucb_rewards.ucb_states.ucb_inited_index
        else:
            log_steps = np.log(steps) if steps > 0 else 0.0
            counts_np = np.array(ucb_rewards.counts, dtype=np.float64)
            q_values = ucb_rewards.q_values

            ucb_values = q_values + np.sqrt(2 * log_steps / np.maximum(counts_np, 1))

            ucb_rewards.ucb_values = ucb_values

            choice = int(np.argmax(ucb_values))

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
