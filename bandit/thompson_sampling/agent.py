"""Thompson Sampling 算法代理类

公开接口：
TSAgent: Thompson Sampling 算法代理类
"""

from __future__ import annotations

from core import BaseAgent, BaseAlgorithm
from core.environment import RLEnv
from .schemas import TSRewardsState, TSAlgorithmType

class TSAgent(BaseAgent[TSRewardsState, "TSAlgorithm"]):
    """TS 算法代理类"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        algorithm: "TSAlgorithm",
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """TS 算法代理类初始化"""
        super().__init__(name, env, algorithm, convergence_threshold, convergence_min_steps, seed)

        self.rewards = TSRewardsState.from_env(env=self.env)

    def act(self, **kwargs) -> int:
        self.steps += 1
        return self.algorithm.run()

    def pull_machine(self, machine_id: int) -> int:
        reward = self.env.pull(machine_id, self.steps)
        self._update_rewards(machine_id, reward)
        self._check_convergence()
        return reward

    def _update_rewards(self, machine_id: int, reward: int):
        ts_rewards = self.rewards
        ts_rewards.counts[machine_id] += 1
        ts_rewards.values[machine_id] += reward

        if reward > 0:
            ts_rewards.alpha[machine_id] += 1
        else:
            ts_rewards.beta[machine_id] += 1

class TSAlgorithm(BaseAlgorithm[TSAgent, TSAlgorithmType]):
    def __init__(self, ts_type: TSAlgorithmType = TSAlgorithmType.TS) -> None:
        super().__init__(ts_type, TSAgent)
        
    def set_agent(self, agent: TSAgent) -> None:
        super().set_agent(agent)
    
    def run(self) -> int:
        return self.ts()
    
    def ts(self) -> int:
        r = self.agent.rewards
        beta = self.agent.rng.beta(r.alpha, r.beta)
        index = int(beta.argmax())
        return index
