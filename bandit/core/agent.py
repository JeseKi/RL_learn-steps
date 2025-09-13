"""核心模块：代理抽象类

公开接口：
- BaseAgent: 代理抽象基类，定义了所有代理必须实现的方法
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from .environment import RLEnv
from .schemas import BaseRewardsState, Metrics


class BaseAgent(ABC):
    """代理抽象基类，所有多臂老虎机代理必须继承这个类"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        convergence_threshold: float = 0.9,
        convergence_min_steps: int = 100,
        seed: int = 42,
    ) -> None:
        """抽象代理基类初始化

        Args:
            name (str): 代理名称
            env (RLEnv): 环境
            convergence_threshold (float): 收敛阈值
            convergence_min_steps (int): 最小收敛步数
            seed (int): 随机种子
        """
        self.name: str = name
        self.seed: int = seed
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.env: RLEnv = env
        self.steps: int = 0
        self.metrics_history: List[Tuple[BaseRewardsState, Metrics, int]] = []
        self.rewards: BaseRewardsState = BaseRewardsState.from_env(env=env, initial_value=0.0, initial_count=0)
        self.convergence_threshold = convergence_threshold
        self.convergence_min_steps = convergence_min_steps
        self.convergence_steps = 0

    @abstractmethod
    def act(self, **kwargs) -> int:
        """选择行动（拉动哪个老虎机）

        Returns:
            int: 选择的机器ID
        """
        self.steps += 1
        raise NotImplementedError("子类必须实现 act 方法")

    @abstractmethod
    def pull_machine(self, machine_id: int) -> int:
        """拉动指定机器并更新状态

        Args:
            machine_id (int): 机器ID

        Returns:
            int: 获得的奖励
        """
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
        return self.rewards.counts[-1] / self.steps

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