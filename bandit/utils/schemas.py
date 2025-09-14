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
