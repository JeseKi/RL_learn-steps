"""
文件功能：
- 过程 JSON 文件读取、按算法归组与公共步长的均值聚合。

公开接口：
- load_process_file(path)
- group_runs_by_agent(paths)
- aggregate_means_by_agent(grouped)

内部方法：
- _intersect_sorted_steps(step_lists)

公开接口的 pydantic 模型：
- 使用 utils.schemas 中的 ProcessRun、AggregatedSeries、AggregatedResult。
"""

from typing import Dict, List, Mapping, Sequence
from pathlib import Path
import json
import numpy as np

from utils.schemas import ProcessRun, AggregatedSeries, AggregatedResult
from utils.plot_const import METRICS_FOR_2x2


def load_process_file(path: Path) -> ProcessRun:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return ProcessRun.model_validate(raw)


def group_runs_by_agent(paths: Sequence[Path]) -> Dict[str, List[ProcessRun]]:
    grouped: Dict[str, List[ProcessRun]] = {}
    for p in paths:
        run = load_process_file(p)
        if not run.points:
            continue
        agent_name = run.points[0].data.agent_name
        grouped.setdefault(agent_name, []).append(run)
    if not grouped:
        raise ValueError("未读取到任何有效的过程数据点")
    return grouped


def _intersect_sorted_steps(step_lists: Sequence[Sequence[int]]) -> List[int]:
    if not step_lists:
        return []
    s = set(step_lists[0])
    for lst in step_lists[1:]:
        s &= set(lst)
    return sorted(s)


def aggregate_means_by_agent(grouped: Mapping[str, Sequence[ProcessRun]]) -> AggregatedResult:
    series_by_metric: Dict[str, List[AggregatedSeries]] = {m: [] for m in METRICS_FOR_2x2}
    for agent_name, runs in grouped.items():
        if not runs:
            continue
        step_lists = [[pt.step for pt in r.points] for r in runs]
        common_steps = _intersect_sorted_steps(step_lists)
        if not common_steps:
            raise ValueError(f"算法 {agent_name} 的多轮过程没有公共步长，无法聚合")
        values_acc: Dict[str, List[float]] = {m: [] for m in METRICS_FOR_2x2}
        for s in common_steps:
            step_vals: Dict[str, List[float]] = {m: [] for m in METRICS_FOR_2x2}
            for r in runs:
                for pt in r.points:
                    if pt.step == s:
                        d = pt.data
                        step_vals["total_reward"].append(float(d.total_reward))
                        step_vals["optimal_rate"].append(float(d.optimal_rate))
                        step_vals["regret"].append(float(d.regret))
                        step_vals["regret_rate"].append(float(d.regret_rate))
                        break
            for m in METRICS_FOR_2x2:
                values_acc[m].append(float(np.mean(step_vals[m])) if step_vals[m] else float("nan"))
        for m in METRICS_FOR_2x2:
            series_by_metric[m].append(
                AggregatedSeries(metric=m, algorithm=agent_name, steps=common_steps, values=values_acc[m]) # type: ignore # 误报
            )
    return AggregatedResult(metrics=series_by_metric)

