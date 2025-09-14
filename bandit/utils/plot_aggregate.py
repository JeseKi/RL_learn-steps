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


def aggregate_means_by_agent(
    grouped: Mapping[str, Sequence[ProcessRun]],
) -> AggregatedResult:
    """对每个算法进行“按步聚合均值”。

    关键修正：
    - 单个 process.json 文件中，同一步 (step) 可能包含来自多个 seed 的多条记录；
      之前实现只取“第一条匹配记录”导致均值偏差。现在会对“同一步内的所有记录”先求均值，
      再在多个文件之间做均值聚合。
    数据流：
    - 输入 grouped: {algorithm -> [ProcessRun, ...]}；
    - 对每个 algorithm：
        1) 计算该算法在所有 runs 的公共步长 common_steps；
        2) 对于每个 step s：
            - 在每个 run 内收集该 step 的所有记录（多 seed），按指标求“run 内均值”；
            - 将各 run 的“run 内均值”再做一次均值，得到该算法在 step s 的最终值；
        3) 输出 AggregatedSeries；
    - 返回 AggregatedResult。
    """
    series_by_metric: Dict[str, List[AggregatedSeries]] = {
        m: [] for m in METRICS_FOR_2x2
    }
    for agent_name, runs in grouped.items():
        if not runs:
            continue
        # 1) 计算公共步长（忽略重复步的多次记录，仅用于集合求交）
        step_lists = [[pt.step for pt in r.points] for r in runs]
        common_steps = _intersect_sorted_steps(step_lists)
        if not common_steps:
            raise ValueError(f"算法 {agent_name} 的多轮过程没有公共步长，无法聚合")

        # 2) 对每个公共步长做两层均值：run 内（多 seed）均值 -> 多 run 均值
        values_acc: Dict[str, List[float]] = {m: [] for m in METRICS_FOR_2x2}
        for s in common_steps:
            # 收集“每个 run 在该步的 run 内均值”
            per_run_means: Dict[str, List[float]] = {m: [] for m in METRICS_FOR_2x2}
            for r in runs:
                # 找到该 run 内所有 step==s 的记录（可能对应多个 seed）
                tr_vals: List[float] = []
                or_vals: List[float] = []
                rg_vals: List[float] = []
                rr_vals: List[float] = []
                for pt in r.points:
                    if pt.step == s:
                        d = pt.data
                        tr_vals.append(float(d.total_reward))
                        or_vals.append(float(d.optimal_rate))
                        rg_vals.append(float(d.regret))
                        rr_vals.append(float(d.regret_rate))
                # 若该 run 在该步没有记录，则跳过（不参与均值）
                if tr_vals:
                    per_run_means["total_reward"].append(float(np.mean(tr_vals)))
                if or_vals:
                    per_run_means["optimal_rate"].append(float(np.mean(or_vals)))
                if rg_vals:
                    per_run_means["regret"].append(float(np.mean(rg_vals)))
                if rr_vals:
                    per_run_means["regret_rate"].append(float(np.mean(rr_vals)))

            # 多 run 再做一次均值
            for m in METRICS_FOR_2x2:
                vals = per_run_means[m]
                values_acc[m].append(float(np.mean(vals)) if vals else float("nan"))

        # 3) 输出序列
        for m in METRICS_FOR_2x2:
            series_by_metric[m].append(
                AggregatedSeries(
                    metric=m,
                    algorithm=agent_name,
                    steps=common_steps,
                    values=values_acc[m],
                )  # type: ignore[call-arg]
            )
    return AggregatedResult(metrics=series_by_metric)
