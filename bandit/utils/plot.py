"""
聚合导出模块（不超过 300 行）：

文件功能：
- 对外统一暴露绘图与 CLI 的公开接口，实际实现分散在子模块：
  - plot_font：字体与 matplotlib 延迟导入
  - plot_aggregate：读取/分组/聚合
  - plot_intersections：交点检测
  - plot_draw：历史指标绘图（兼容）
  - plot_cli：2x2 绘图与 CLI

公开接口：
- plot_metrics_history, load_process_file, group_runs_by_agent, aggregate_means_by_agent,
  find_pairwise_intersections_for_metric, find_axis_intersections_for_series, cli_main

内部方法：
- 无（仅转发）。

公开接口的 pydantic 模型：
- 见 utils/schemas.py。
"""

from utils.plot_font import _ensure_matplotlib
from utils.plot_const import METRICS_FOR_2x2
from utils.plot_draw import plot_metrics_history
from utils.plot_aggregate import (
    load_process_file,
    group_runs_by_agent,
    aggregate_means_by_agent,
)
from utils.plot_intersections import (
    find_pairwise_intersections_for_metric,
    find_axis_intersections_for_series,
)
from utils.plot_cli import cli_main

__all__ = [
    "_ensure_matplotlib",
    "METRICS_FOR_2x2",
    "plot_metrics_history",
    "load_process_file",
    "group_runs_by_agent",
    "aggregate_means_by_agent",
    "find_pairwise_intersections_for_metric",
    "find_axis_intersections_for_series",
    "cli_main",
]
