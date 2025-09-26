"""代理抽象基类和算法抽象基类"""

from __future__ import annotations

from typing import List, TypeVar, Generic, Type, Tuple
from abc import ABC, abstractmethod
import numpy as np

from .schemas import BaseRewardsState, BaseAlgorithmType
from .environment import RLEnv
from .schemas import Metrics


AgentRewardState_T = TypeVar("AgentRewardState_T", bound=BaseRewardsState)
AgentAlgorithm_T = TypeVar("AgentAlgorithm_T", bound="BaseAlgorithm")
AlgorithmAgent_T = TypeVar("AlgorithmAgent_T", bound="BaseAgent")
AlgorithmType_T = TypeVar("AlgorithmType_T", bound=BaseAlgorithmType)


class BaseAlgorithm(ABC, Generic[AlgorithmAgent_T, AlgorithmType_T]):
    """算法抽象基类"""

    def __init__(
        self, algorithm_type: AlgorithmType_T, target_agent_type: Type[AlgorithmAgent_T]
    ) -> None:
        self._agent: AlgorithmAgent_T
        self._target_type: Type[AlgorithmAgent_T] = target_agent_type
        self.algorithm_type: AlgorithmType_T = algorithm_type

    @property
    def agent(self) -> AlgorithmAgent_T:
        return self._agent

    @abstractmethod
    def set_agent(self, agent: AlgorithmAgent_T) -> None:
        if not isinstance(agent, self._target_type):
            raise ValueError(f"Agent 必须是 {self._target_type} 类型")
        self._agent = agent

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError("子类必须实现 run 方法")


class BaseAgent(ABC, Generic[AgentRewardState_T, AgentAlgorithm_T]):
    """代理抽象基类，所有多臂老虎机代理必须继承这个类"""

    def __init__(
        self,
        name: str,
        env: "RLEnv",
        algorithm: AgentAlgorithm_T,
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """抽象代理基类初始化"""
        self.rewards: AgentRewardState_T

        self.name: str = name
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.env: "RLEnv" = env
        self.algorithm: AgentAlgorithm_T = algorithm
        self.algorithm.set_agent(self)

        self.steps: int = 0
        self.metrics_history: List[Tuple[BaseRewardsState, Metrics, int]] = []
        self.convergence_threshold = convergence_threshold
        self.convergence_min_steps = convergence_min_steps
        self.convergence_steps = 0

    @abstractmethod
    def act(self, **kwargs) -> int:
        """选择行动（拉动哪个老虎机）"""
        self.steps += 1
        raise NotImplementedError("子类必须实现 act 方法")

    @abstractmethod
    def pull_machine(self, machine_id: int) -> int:
        """拉动指定机器并更新状态"""
        raise NotImplementedError("子类必须实现 pull_machine 方法")

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
        return self.rewards.counts[self.env.best_machine_index] / self.steps

    def metric(self) -> Metrics:
        """获取当前指标"""
        return Metrics(
            regret=self.regret(),
            regret_rate=self.regret_rate(),
            rewards=self.rewards.model_copy(deep=True),
            optimal_rate=self.optimal_rate(),
        )

    def _check_convergence(self):
        """检查是否达到收敛条件"""
        if self.steps < self.convergence_min_steps or self.convergence_steps > 0:
            return

        if self.optimal_rate() >= self.convergence_threshold:
            self.convergence_steps = self.steps
            print(f"达到收敛时的步数: {self.convergence_steps}")
            return

    def _check_static_convergence(self):
        """检查是否达到静态收敛条件"""
        if self.steps < self.convergence_min_steps or self.convergence_steps > 0:
            return

        if self.optimal_rate() >= self.convergence_threshold:
            self.convergence_steps = self.steps
            print(f"达到收敛时的步数: {self.convergence_steps}")
            return
        
    def check_dynamic_convergence(self):
        pass