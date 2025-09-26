"""数据库模型：包含Pydantic数据模型

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
        env: "RLEnv",
        initial_value: float = 0.0,
        initial_count: int = 0,
    ) -> BaseRewardsState:
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
        )


class BaseAlgorithmType(str, Enum):
    """算法类型"""

    pass


class Metrics(BaseModel):
    regret: float = Field(..., description="后悔值")
    regret_rate: float = Field(..., description="后悔率")
    rewards: BaseRewardsState = Field(..., description="奖励")
    optimal_rate: float = Field(..., description="最佳臂命中率")

class DynamicConvergenceConfig(BaseModel):
    """动态收敛配置"""

    threshold: float = Field(..., description="判断收敛阈值。最佳臂命中率大于这个阈值时认为收敛")
    min_step: int = Field(..., description="最低收敛步数。达到收敛命中率后，至少要保持这个命中率达到这个步数才能算作收敛")
    smooth_window: int = Field(..., description="平滑窗口。用于计算平滑后的最佳臂命中率")
    W: int = Field(..., description="")