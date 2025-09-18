"""贪婪算法模块：贪婪算法代理类

公开接口：
- GreedyAgent: 贪婪算法代理类，继承自 BaseAgent
"""

from __future__ import annotations

from core import BaseAgent, BaseAlgorithm
from core.environment import RLEnv
from .schemas import GreedyRewardsState, GreedyType
from .config import EpsilonDecreasingConfig, EpsilonDecreasingState


class GreedyAgent(BaseAgent[GreedyRewardsState, "GreedyAlgorithm"]):
    """贪婪算法代理类，继承自 BaseAgent"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        algorithm: "GreedyAlgorithm",
        epsilon_config: EpsilonDecreasingConfig = EpsilonDecreasingConfig(),
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """贪婪算法代理初始化

        Args:
            name (str): 名称
            env (RLEnv): 环境
            algorithm (GreedyAlgorithm): 所使用的贪婪算法
            epsilon_config (EpsilonDecreasingConfig, optional): 退火配置
            optimistic_init (bool, optional): 是否使用乐观初始化
            optimistic_times (int, optional): 乐观初始化的次数
            convergence_threshold (float, optional): 达到收敛条件的阈值（最佳臂选择率）
            convergence_min_steps (int, optional): 达到收敛条件的最小次数，至少要达到这个次数才能算作收敛
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

        self.episode_state = EpsilonDecreasingState(
            epsilon=epsilon_config.start_epsilon,
            decay=epsilon_config.decay,
            min_epsilon=epsilon_config.min_epsilon,
        )
        self.rewards = GreedyRewardsState.from_env(
            env=env,
            optimistic_init=algorithm.optimistic_init,
            optimistic_times=algorithm.optimistic_times,
        )

    def act(self, **kwargs) -> int:
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
        rewards = self.rewards
        rewards.counts[machine_id] += 1
        count = rewards.counts[machine_id]
        rewards.values[machine_id] += reward

        # Q(A) ← Q(A) + (R - Q(A)) / N(A)
        old_q = rewards.q_values[machine_id]
        rewards.q_values[machine_id] = old_q + (reward - old_q) / count


class GreedyAlgorithm(BaseAlgorithm[GreedyAgent, GreedyType]):
    def __init__(
        self,
        greedy_type: GreedyType,
        optimistic_init: bool = False,
        optimistic_times: int = 1,
    ) -> None:
        super().__init__(greedy_type, GreedyAgent)
        self.optimistic_init = optimistic_init
        self.optimistic_times = optimistic_times
        self.optimistic_inited = False

    def set_agent(self, agent: GreedyAgent) -> None:
        super().set_agent(agent)

    def run(self) -> int:
        if self.optimistic_init and not self.optimistic_inited:
            machine_id: int
            for r in self.agent.rewards.q_values_optimistic:
                if r > 0:
                    machine_id = self.agent.rewards.q_values_optimistic.index(r)
                    self.agent.rewards.q_values_optimistic[machine_id] -= 1
                    return machine_id

            self.optimistic_inited = True

        if self.algorithm_type == GreedyType.GREEDY:
            return self.greedy()
        elif self.algorithm_type == GreedyType.EPSILON:
            return self.epsilon()
        elif self.algorithm_type == GreedyType.EPSILON_DECREASING:
            return self.epsilon_decreasing()
        elif self.algorithm_type == GreedyType.GREEDY_ACCUMULATED:
            return self.greedy_accumulated()
        elif self.algorithm_type == GreedyType.EPSILON_ACCUMULATED:
            return self.epsilon_accumulated()
        elif self.algorithm_type == GreedyType.EPSILON_DECREASING_ACCUMULATED:
            return self.epsilon_decreasing_accumulated()
        else:
            raise ValueError(f"不支持的贪婪算法类型: {self.algorithm_type}")

    # Q值贪婪算法
    def greedy(self) -> int:
        return self.agent.rewards.q_values.index(max(self.agent.rewards.q_values))

    def epsilon(self) -> int:
        agent = self.agent
        if agent.rng.random() < agent.episode_state.epsilon:
            return int(agent.rng.integers(0, len(agent.rewards.q_values)))
        else:
            return agent.rewards.q_values.index(max(agent.rewards.q_values))

    def epsilon_decreasing(self) -> int:
        agent = self.agent
        if agent.rng.random() < agent.episode_state.epsilon:
            action = int(agent.rng.integers(0, len(agent.rewards.q_values)))
        else:
            action = agent.rewards.q_values.index(max(agent.rewards.q_values))

        agent.episode_state.epsilon = max(
            agent.episode_state.min_epsilon,
            agent.episode_state.epsilon * agent.episode_state.decay,
        )
        return int(action)

    # 累计奖励贪婪算法
    def greedy_accumulated(self) -> int:
        return self.agent.rewards.values.index(max(self.agent.rewards.values))

    def epsilon_accumulated(self) -> int:
        agent = self.agent
        if agent.rng.random() < agent.episode_state.epsilon:
            return int(agent.rng.integers(0, len(agent.rewards.values)))
        else:
            return agent.rewards.values.index(max(agent.rewards.values))

    def epsilon_decreasing_accumulated(self) -> int:
        """ε-递减贪婪算法：ε 随时间递减，其余时间选累计奖励最高的"""
        agent = self.agent
        if agent.rng.random() < agent.episode_state.epsilon:
            action = int(agent.rng.integers(0, len(agent.rewards.values)))
        else:
            action = agent.rewards.values.index(max(agent.rewards.values))

        agent.episode_state.epsilon = max(
            agent.episode_state.min_epsilon,
            agent.episode_state.epsilon * agent.episode_state.decay,
        )
        return int(action)
