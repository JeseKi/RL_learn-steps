"""
文件功能：
- 提供“实验过程数据”的按对数网格采样与保存能力；
- 保持与“指标/结果保存”解耦（过程数据与指标分开保存）。

公开接口：
- ProcessDataLogger：主类，基于对数网格 t_grid 采样、累积过程数据点，并保存为 .json 文件；
- save_experiment_data：用于保存实验最终结果（奖励与平均指标）的工具函数（与过程数据分开）。

内部方法：
- _normalize_step(step: int) -> int：步数归一化，确保最小为 1；
- _build_log_grid(total_steps: int, n: int) -> list[int]：t_grid=unique(ceil(10^{linspace(0,log10 T,N)}))。

公开接口的 Pydantic 模型：
- utils.schemas.ProcessDataPoint：表示单条过程数据点；
- utils.schemas.ProcessDataDump：表示包含元信息与数据点的文件模型。

说明：
- 一个文件只包含一个主要功能类：本文件的主要功能类为 ProcessDataLogger；
- 过程数据保存与指标保存（save_experiment_data）分离，互不影响。
"""

from __future__ import annotations

import json
from pathlib import Path
import math
from typing import Any, Dict, List, Optional

from core.schemas import BaseRewardsState
from train import AverageMetrics
from utils.schemas import ProcessDataDump, ProcessDataPoint


def _normalize_step(step: int) -> int:
    """内部方法：步数归一化，至少为 1。"""
    return 1 if step <= 0 else int(step)


def _build_log_grid(total_steps: int, n: int) -> List[int]:
    """内部方法：构造对数网格 t_grid

    公式：t_grid = unique( ceil( 10^{ linspace(0, log10(T), N) } ) )
    - 始终包含 1 与 T；
    - 返回严格递增的唯一序列。
    """
    if total_steps <= 0:
        raise ValueError("total_steps 必须为正整数")
    if n <= 0:
        raise ValueError("grid_size(N) 必须为正整数")

    logT = 0.0 if total_steps == 1 else math.log10(total_steps)
    if n == 1:
        raw = [0.0]
    else:
        step = logT / (n - 1)
        raw = [i * step for i in range(n)]

    grid = [int(math.ceil(10**v)) for v in raw]
    grid.append(1)
    grid.append(int(total_steps))

    grid = sorted(set(grid))
    grid = [x for x in grid if 1 <= x <= total_steps]
    return grid


class ProcessDataLogger:
    """实验过程数据记录器

    公开接口：
    - grid: 属性，返回当前对数采样网格（只读副本）；
    - should_record(step: int) -> bool：给定步数是否应当记录；
    - add(step: int, data: Dict[str, Any]) -> None：添加一条过程数据（若外部自行判断则可忽略 should_record）；
    - export(total_steps: Optional[int] = None) -> ProcessDataDump：导出为 Pydantic 模型；
    - save(file_name: Path | str, total_steps: Optional[int] = None) -> Path：保存为 .json 文件。

    数据流预期：
    - 外部训练循环每步调用 should_record(step) 判断是否采样；
    - 若 True，则调用 add(step, data) 将该步的过程数据（键值对）入队；
    - 完成后调用 save(...) 将全部过程数据一次性写入 .json 文件；
    - 该 .json 仅包含过程数据，不包含指标与最终聚合结果。
    """

    def __init__(
        self,
        run_id: str,
        total_steps: int,
        grid_size: int = 100,
    ) -> None:
        self.run_id = run_id
        self._points: List[ProcessDataPoint] = []
        self._grid: Optional[List[int]] = None
        self._grid_set: Optional[set[int]] = None
        self._grid_size = grid_size
        
        self._set_total_steps(total_steps)

    @property
    def grid(self) -> List[int]:
        """公开接口：返回当前对数采样网格（只读副本）。"""
        return list(self._grid) if self._grid is not None else []

    def should_record(self, step: int) -> bool:
        """是否在当前步数记录一次数据。

        逻辑：使用对数网格 t_grid，当前步数 s 在网格中则记录。
        """
        if self._grid_set is None:
            raise RuntimeError("未设置 total_steps，无法进行网格采样。请在构造时传入或调用 set_total_steps(total_steps)。")
        s = _normalize_step(step)
        return s in self._grid_set

    def add(self, step: int, data: Dict[str, Any]) -> None:
        """加入一条过程数据点（不做去重，由调用方控制采样节奏）。"""
        s = _normalize_step(step)
        self._points.append(ProcessDataPoint(step=s, data=data))

    def export(self, total_steps: int) -> ProcessDataDump:
        """导出全部过程数据为 Pydantic 模型。"""
        return ProcessDataDump(
            run_id=self.run_id,
            total_steps=total_steps,
            points=self._points,
        )

    def save(self, file_name: Path | str, total_steps: int) -> Path:
        """将过程数据保存为 .json 文件（UTF-8, 缩进 4）。"""
        path = Path(file_name)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".json":
            path = path.with_suffix(".json")

        dump_model = self.export(total_steps=total_steps)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dump_model.model_dump(), f, ensure_ascii=False, indent=4)

        print(f"✅ 过程数据已保存至 {path}")
        return path

    def _set_total_steps(self, total_steps: int) -> None:
        """设置/更新总步数，并重建对数网格。"""
        grid = _build_log_grid(total_steps=total_steps, n=self._grid_size)
        self._grid = grid
        self._grid_set = set(grid)


def save_experiment_data(
    reward: BaseRewardsState, metrics: AverageMetrics, file_name: Path
):
    """保存实验最终结果（奖励 + 平均指标）为 JSON 文件。

    注意：该函数与“过程数据保存”解耦；若需保存过程数据，请使用 ProcessDataLogger。
    """

    experiment_data = {
        "reward": reward.model_dump(),
        "metrics": metrics.model_dump(),
    }

    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True, exist_ok=True)
    if file_name.suffix != ".json":
        file_name = file_name.with_suffix(".json")
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 实验结果数据已保存至 {file_name}")
