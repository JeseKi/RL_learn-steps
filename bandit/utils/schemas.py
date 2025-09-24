"""
文件功能：
- 定义 utils 模块内用于“实验过程数据记录”和“绘图聚合结果/标注点”的 Pydantic 数据模型。

公开接口：
- 过程数据模型：
  - ProcessDataPoint：单条过程数据点的模型。
  - ProcessDataDump：过程数据文件的整体模型（包含元信息与数据点列表）。
- 绘图/聚合模型：
  - ProcessPointData / ProcessPoint / ProcessRun：用于解析来自日志的过程数据（更强结构化）。
  - AggregatedSeries / AggregatedResult：用于表达聚合后的曲线数据。
  - IntersectionPoint / AxisIntersection：用于表达图上需要标注的交点信息。

内部方法：
- 无（本文件仅包含数据模型定义）。

公开接口的 Pydantic 模型：
- 本文件中定义的所有类均为 Pydantic BaseModel 子类，可直接作为公开接口返回值。

说明：
- 一个文件只包含一个主要功能：本文件专注于“与过程数据与绘图聚合相关的数据模型定义”。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class ExperimentMeta(BaseModel):
    """实验元信息数据模型

    字段：
    1. 运行日期 (run_date)
    2. 运行 id (run_id)
    3. 总步数 (total_steps)
    4. 臂数量 (num_arms)
    5. Agent 初始种子 (agent_seed)
    6. Agent 算法名称 (agent_algorithm)
    7. Agent 重复运行次数 (agent_runs)
    8. 收敛阈值 (convergence_threshold)
    9. 最低收敛步数 (min_convergence_steps)
    10. 是否启用乐观初始化 (optimistic_initialization_enabled)
    11. 乐观初始化次数 (optimistic_initialization_value)
    12. 常数步长 (constant_alpha)
    13. 折扣因子 (discount_factor)
    14. 环境静态/动态 (environment_type)
    15. 环境动态变化方法 (environment_dynamic_method)
        1. 随机漫步间隔 (random_walk_interval)
        2. 每次随机漫步所影响的机器数量 (random_walk_affected_arms)
        3. 分段平稳间隔 (piecewise_stationary_interval)
        4. 分段平稳方法 (piecewise_stationary_method)
        5. 环境种子 (environment_seed)
    """

    run_date: str = Field(..., description="运行日期")
    run_id: str = Field(..., description="运行 ID")
    total_steps: int = Field(..., ge=1, description="总步数")
    num_arms: int = Field(..., ge=1, description="臂数量")
    agent_seed: Optional[int] = Field(default=None, description="Agent 初始种子")
    agent_algorithm: str = Field(..., description="Agent 算法名称")
    agent_runs: int = Field(..., ge=1, description="Agent 重复运行次数")
    convergence_threshold: float = Field(..., description="收敛阈值")
    min_convergence_steps: int = Field(..., ge=1, description="最低收敛步数")
    optimistic_initialization_enabled: bool = Field(
        ..., description="是否启用乐观初始化"
    )
    optimistic_initialization_value: Optional[float] = Field(
        default=None, description="乐观初始化值"
    )
    constant_alpha: Optional[float] = Field(default=None, description="常数步长")
    discount_factor: Optional[float] = Field(default=None, description="折扣因子")
    environment_type: str = Field(..., description="环境类型 (静态/动态)")
    environment_dynamic_method: Optional[str] = Field(
        default=None, description="环境动态变化方法"
    )
    # 环境动态变化方法的参数
    random_walk_interval: Optional[int] = Field(
        default=None, ge=1, description="随机漫步间隔"
    )
    random_walk_affected_arms: Optional[int] = Field(
        default=None, ge=1, description="每次随机漫步所影响的机器数量"
    )
    piecewise_stationary_interval: Optional[int] = Field(
        default=None, ge=1, description="分段平稳间隔"
    )
    piecewise_stationary_method: Optional[str] = Field(
        default=None, description="分段平稳方法"
    )
    environment_seed: Optional[int] = Field(default=None, description="环境种子")


class ProcessDataPoint(BaseModel):
    """单条实验过程数据点

    字段：
    - step: 当前全局步数（从 1 开始）。
    - data: 与该步对应的任意键值内容（如动作、奖励、内部状态摘要等）。
    """

    step: int = Field(..., ge=1, description="当前步数（>=1）")
    data: Dict[str, Any] = Field(default_factory=dict, description="任意键值内容")


class ProcessDataDump(BaseModel):
    """实验过程数据文件模型

    字段：
    - run_id: 运行 ID 或名称，用于区分不同实验。
    - created_at: 导出文件时间（ISO 格式）。
    - total_steps: 实验总步数（可选）。
    - points: 过程数据点列表。
    """

    run_id: Optional[str] = Field(default=None, description="运行 ID/名称")
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    total_steps: Optional[int] = Field(default=None, ge=1, description="实验总步数")
    points: List[ProcessDataPoint] = Field(default_factory=list)


# ------- 绘图/聚合模型（用于 utils.plot 的公开接口） -------


class ProcessPointData(BaseModel):
    """单个 step 的结构化指标数据。"""

    agent_name: str
    seed: Optional[int] = None
    regret: float
    regret_rate: float
    total_reward: float
    optimal_rate: float
    convergence_steps: int = Field(default=0)


class ProcessPoint(BaseModel):
    """过程数据中的一个记录点（结构化）。"""

    step: int
    data: ProcessPointData


class ProcessRun(BaseModel):
    """单个过程文件（一次运行或一次记录）的整体结构。"""

    run_id: Optional[str] = None
    created_at: Optional[str] = None
    total_steps: Optional[int] = None
    points: List[ProcessPoint]


class AggregatedSeries(BaseModel):
    """聚合后的单条曲线。"""

    metric: Literal["total_reward", "optimal_rate", "regret", "regret_rate"]
    algorithm: str
    steps: List[int]
    values: List[float]


class AggregatedResult(BaseModel):
    """聚合结果：每个指标下包含若干算法的曲线。"""

    metrics: Dict[str, List[AggregatedSeries]]


class IntersectionPoint(BaseModel):
    """两条曲线的交点信息（用于标注）。"""

    x: float
    y: float
    algo_a: str
    algo_b: str
    metric: str


class AxisIntersection(BaseModel):
    """曲线与坐标轴的近似交点信息（用于标注）。"""

    axis: Literal["x", "y"]
    x: float
    y: float
    algorithm: str
    metric: str
