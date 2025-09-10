"""核心模块：代理抽象类

公开接口：
- BaseAgent: 代理抽象基类，定义了所有代理必须实现的方法
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import random

from .environment import RLEnv
from .schemas import RewardsState, Metrics


class BaseAgent(ABC):
    """代理抽象基类，所有多臂老虎机代理必须继承这个类"""

    def __init__(
        self,
        name: str,
        env: RLEnv,
        seed: int = 42,
    ) -> None:
        """抽象代理基类初始化

        Args:
            name (str): 代理名称
            env (RLEnv): 环境
            seed (int): 随机种子
        """
        self.name = name
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.env = env
        self.steps: int = 0
        self.metrics_history: List[Tuple[RewardsState, Metrics, int]] = []
        self.rewards: RewardsState = RewardsState.from_env(env)

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

    @abstractmethod
    def metric(self) -> Metrics:
        """获取当前指标

        Returns:
            Metrics: 当前评估指标
        """
        raise NotImplementedError("子类必须实现 metric 方法")

    @abstractmethod
    def regret(self) -> float:
        """计算后悔值

        Returns:
            float: 后悔值
        """
        raise NotImplementedError("子类必须实现 regret 方法")

    @abstractmethod
    def regret_rate(self) -> float:
        """计算后悔率

        Returns:
            float: 后悔率
        """
        raise NotImplementedError("子类必须实现 regret_rate 方法")

    @abstractmethod
    def optimal_rate(self) -> float:
        """计算最优臂选择率

        Returns:
            float: 最优臂选择率
        """
        raise NotImplementedError("子类必须实现 optimal_rate 方法")