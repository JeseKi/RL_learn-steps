"""
文件功能：
- 定义 utils 模块内用于“实验过程数据记录”的 Pydantic 数据模型。

公开接口：
- ProcessDataPoint：单条过程数据点的模型。
- ProcessDataDump：过程数据文件的整体模型（包含元信息与数据点列表）。

内部方法：
- 无（本文件仅包含数据模型定义）。

公开接口的 Pydantic 模型：
- 本文件中定义的所有类均为 Pydantic BaseModel 子类，可直接作为公开接口返回值。

说明：
- 一个文件只包含一个主要功能：本文件专注于“过程数据的数据模型定义”。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

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
