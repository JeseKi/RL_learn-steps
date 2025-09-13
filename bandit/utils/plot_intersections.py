"""
文件功能：
- 曲线交点与坐标轴交点的检测与插值计算。

公开接口：
- find_pairwise_intersections_for_metric(steps, series_by_algo, min_gap_ratio=0.1)
- find_axis_intersections_for_series(steps, values)

内部方法：
- _linear_interp(x0, y0, x1, y1, y=0.0)

公开接口的 pydantic 模型：
- 使用 utils.schemas 中的 IntersectionPoint 与 AxisIntersection。
"""

from typing import Mapping, Sequence, List
import math
import itertools
import numpy as np

from utils.schemas import IntersectionPoint, AxisIntersection


def _linear_interp(x0: float, y0: float, x1: float, y1: float, y: float = 0.0):
    """在线性段 (x0,y0)-(x1,y1) 上计算与 y 水平线的交点 (x*, y)。"""
    if x1 == x0 or y1 == y0:
        return x0, y
    t = (y - y0) / (y1 - y0)
    x = x0 + t * (x1 - x0)
    return x, y


def find_pairwise_intersections_for_metric(
    steps: Sequence[int],
    series_by_algo: Mapping[str, Sequence[float]],
    *,
    min_gap_ratio: float = 0.1,
) -> List[IntersectionPoint]:
    """计算同一指标下不同算法曲线的交点，并按最小间距过滤。"""
    if not steps:
        return []
    n = len(steps)
    min_gap = max(1, int(math.ceil(n * float(min_gap_ratio))))
    algos = list(series_by_algo.keys())
    results: List[IntersectionPoint] = []
    for a, b in itertools.combinations(algos, 2):
        ya = series_by_algo[a]
        yb = series_by_algo[b]
        if len(ya) != n or len(yb) != n:
            continue
        last_mark_idx = -min_gap * 2
        for i in range(n - 1):
            d0 = ya[i] - yb[i]
            d1 = ya[i + 1] - yb[i + 1]
            if (d0 == 0.0) or (d0 * d1 < 0):
                x0, x1 = float(steps[i]), float(steps[i + 1])
                dy_a = ya[i + 1] - ya[i]
                dd = (d1 - d0)
                t = 0.0 if dd == 0 else (-d0 / dd)
                t = max(0.0, min(1.0, t))
                x_cross = x0 + t * (x1 - x0)
                y_cross = ya[i] + t * dy_a
                if i - last_mark_idx >= min_gap:
                    results.append(IntersectionPoint(x=x_cross, y=y_cross, algo_a=a, algo_b=b, metric=""))
                    last_mark_idx = i
    return results


def find_axis_intersections_for_series(
    steps: Sequence[int],
    values: Sequence[float],
) -> List[AxisIntersection]:
    """查找单条曲线与 X/Y 轴的近似交点（1% 容差与附加规则）。"""
    if not steps:
        return []
    x_min, x_max = float(steps[0]), float(steps[-1])
    x_range = max(1.0, x_max - x_min)
    y_vals = list(map(float, values))
    y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
    y_range = max(1e-12, y_max - y_min)
    results: List[AxisIntersection] = []
    # Y 轴（近左边界）
    left_dist = float(steps[0]) - x_min
    if left_dist <= 0.01 * x_range:
        results.append(AxisIntersection(axis="y", x=float(steps[0]), y=y_vals[0], algorithm="", metric=""))
    # X 轴（y≈0），且后续值需低于初始值
    if min(y_vals[1:] or [y_vals[0]]) < y_vals[0]:
        threshold = 0.01 * y_range
        for i in range(len(steps) - 1):
            y0, y1 = y_vals[i], y_vals[i + 1]
            if (y0 == 0.0) or (y0 * y1 < 0):
                x0, x1 = float(steps[i]), float(steps[i + 1])
                x_cross, y_cross = _linear_interp(x0, y0, x1, y1, y=0.0)
                if abs(y_cross - 0.0) <= threshold:
                    results.append(AxisIntersection(axis="x", x=x_cross, y=y_cross, algorithm="", metric=""))
                    break
    return results

