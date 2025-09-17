"""核心模块：包含Pydantic数据模型

公开接口：
- BaseRewardsState: 基础奖励状态模型
- Metrics: 评估指标模型
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .environment import RLEnv


class PiecewizeMethod(Enum):
    """分段方法"""

    UPSIDE_DOWN = "upside_down"
    RESET = "reset"
    ADDITION_SUBTRACTION = "addition_subtraction"

class BaseRewardsState(BaseModel):
    """基础奖励状态模型"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: List[float] = Field(description="每个机器的累积奖励")
    counts: List[float] = Field(description="每个机器的拉动次数")

    @classmethod
    def from_env(
        cls,
        env: RLEnv,
        initial_value: float = 0.0,
        initial_count: int = 0,
    ) -> BaseRewardsState:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env: 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (int, optional): counts 列表中每个元素的初始值，默认为 0。

        Returns:
            BaseRewardsState: 一个初始化好的 BaseRewardsState 实例。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
        )


class Metrics(BaseModel):
    regret: float = Field(..., description="后悔值")
    regret_rate: float = Field(..., description="后悔率")
    rewards: BaseRewardsState = Field(..., description="奖励")
    optimal_rate: float = Field(..., description="最佳臂命中率")
