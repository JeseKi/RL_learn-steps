"""Thompson Sampling 算法代理类

公开接口：
TSAgent: Thompson Sampling 算法代理类
"""

from __future__ import annotations
from typing import cast

from core.agent import BaseAgent
from core.environment import RLEnv

from .schemas import TSRewardsState


class TSAgent(BaseAgent):
    """TS 算法代理类"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """TS 算法代理类初始化

        Args:
            name (str): Agent 名称
            env (RLEnv): Agent 环境
            convergence_threshold (float, optional): 收敛阈值
            convergence_min_steps (int, optional): 最小收敛部署，避免波动
            seed (int, optional): 种子
        """

        super().__init__(name, env, convergence_threshold, convergence_min_steps, seed)

        self.rewards = TSRewardsState.from_env(env=self.env)

    def act(self, **kwargs) -> int:
        ts_rewards = cast(TSRewardsState, self.rewards)
        self.steps += 1
        return ts_rewards.get_best_machine(self.rng)

    def pull_machine(self, machine_id: int) -> int:
        reward = self.env.pull(machine_id)
        self._update_rewards(machine_id, reward)
        self._check_convergence()
        return reward

    def _update_rewards(self, machine_id: int, reward: int):
        ts_rewards = cast(TSRewardsState, self.rewards)
        ts_rewards.counts[machine_id] += 1
        ts_rewards.values[machine_id] += reward

        if reward > 0:
            ts_rewards.alpha[machine_id] += 1
        else:
            ts_rewards.beta[machine_id] += 1
