"""TS 算法模块

公开接口：TSRewardsStaet: TS 算法奖励状态模型
"""

from __future__ import annotations

from pydantic import Field
import numpy as np

from core.schemas import BaseRewardsState, BaseAlgorithmType
from core.environment import RLEnv


class TSRewardsState(BaseRewardsState):
    alpha: np.ndarray = Field(..., description="每个机器的 alpha 参数")
    beta: np.ndarray = Field(..., description="每个机器的 beta 参数")

    @classmethod
    def from_env(
        cls: type[TSRewardsState],
        env: RLEnv,
        initial_value: float = 0.0,
        initial_count: int = 0,
    ) -> TSRewardsState:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env: 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (int, optional): counts 列表中每个元素的初始值，默认为 0。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
            alpha=np.array([1] * num_machines, dtype=np.float64),
            beta=np.array([1] * num_machines, dtype=np.float64),
        )

class TSAlgorithmType(BaseAlgorithmType):
    """TS 算法类型"""

    TS = "ts"