"""
文件功能：
- 2x2 指标绘图与命令行入口。

公开接口：
- cli_main(argv=None)

内部方法：
- _plot_2x2(aggregated, ...)

公开接口的 pydantic 模型：
- 使用 utils.schemas 中的 AggregatedResult。
"""

from typing import Dict, Optional, Sequence
from pathlib import Path
import numpy as np

from utils.schemas import AggregatedResult, AggregatedSeries
from utils.plot_const import METRICS_FOR_2x2
from utils.plot_font import _ensure_matplotlib
from utils.plot_intersections import (
    find_axis_intersections_for_series,
    find_pairwise_intersections_for_metric,
)
from utils.plot_aggregate import group_runs_by_agent, aggregate_means_by_agent


def _plot_2x2(
    aggregated: AggregatedResult,
    *,
    title: Optional[str],
    out_file: Path,
    min_cross_gap_ratio: float = 0.1,
    show: bool = False,
    debug: bool = False,
    x_log: bool = False,
) -> None:
    plt, font_prop, title_font_prop = _ensure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
    axes = axes.flatten()
    if title:
        fig.suptitle(title, fontproperties=title_font_prop)
    metric_titles = {
        "total_reward": "累积奖励",
        "optimal_rate": "最佳臂命中率",
        "regret": "后悔值",
        "regret_rate": "后悔率",
    }
    import itertools as _it
    colors = [
        "#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F","#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC",
    ]
    color_cycle = _it.cycle(colors)
    for ax, metric in zip(axes, METRICS_FOR_2x2):
        series_list = aggregated.metrics.get(metric, [])
        if not series_list:
            ax.set_visible(False)
            continue
        algo_to_series: Dict[str, AggregatedSeries] = {s.algorithm: s for s in series_list}
        steps = series_list[0].steps
        # 轴尺度
        if x_log:
            try:
                ax.set_xscale("log", base=10)
            except TypeError:
                # 兼容旧版本 matplotlib
                ax.set_xscale("log")

        algo_to_color: Dict[str, str] = {}
        for algo, s in algo_to_series.items():
            c = next(color_cycle)
            algo_to_color[algo] = c
            ax.plot(s.steps, s.values, label=algo, color=c, linewidth=1.8)
            axis_marks = find_axis_intersections_for_series(s.steps, s.values)
            for mk in axis_marks:
                ax.scatter([mk.x], [mk.y], color=c, s=20, zorder=3)
                if mk.axis == "y":
                    ax.annotate("Y轴附近", (mk.x, mk.y), textcoords="offset points", xytext=(5, 5), fontsize=9, color=c, fontproperties=font_prop)
                else:
                    ax.annotate("X轴交点", (mk.x, mk.y), textcoords="offset points", xytext=(5, -10), fontsize=9, color=c, fontproperties=font_prop)
        if metric == "optimal_rate":
            ax.axhline(0.9, color="#999999", linestyle=":", linewidth=1.2, label="收敛阈值 90%")
            for algo, s in algo_to_series.items():
                arr = np.array(s.values, dtype=float)
                idxs = np.where(arr >= 0.9)[0]
                if idxs.size > 0:
                    i = int(idxs[0])
                    ax.scatter([s.steps[i]], [arr[i]], color=algo_to_color[algo], s=30, zorder=3)
                    ax.annotate(f"收敛@{s.steps[i]}", (s.steps[i], float(arr[i])), textcoords="offset points", xytext=(6, -12), fontsize=9, color=algo_to_color[algo], fontproperties=font_prop)
        series_by_algo = {algo: s.values for algo, s in algo_to_series.items()}
        crosses = find_pairwise_intersections_for_metric(steps, series_by_algo, min_gap_ratio=min_cross_gap_ratio)
        for cp in crosses:
            ax.scatter([cp.x], [cp.y], color="#444444", s=16, zorder=3)
            ax.annotate(f"交点@{cp.x:.0f}", (cp.x, cp.y), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#444", fontproperties=font_prop)
        ax.set_title(metric_titles.get(metric, metric), fontproperties=font_prop)
        ax.set_xlabel("步数", fontproperties=font_prop)
        ax.set_ylabel("值", fontproperties=font_prop)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(prop=font_prop, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_file = out_file.with_suffix(".png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file)
    if show:  # pragma: no cover
        plt.show()


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="过程数据 2x2 指标对比绘图（均值聚合，交点/收敛标注）")
    parser.add_argument("files", nargs="+", help="一个或多个 *process.json 文件路径")
    parser.add_argument("-o", "--out", default="experiment_data/metrics_compare.png", help="输出图片路径")
    parser.add_argument("--title", default=None, help="图表标题")
    parser.add_argument("--min-cross-gap", type=float, default=0.1, help="交点最小间距占比（默认 0.1）")
    parser.add_argument("--show", action="store_true", help="绘制后显示图像（默认不显示）")
    parser.add_argument("--x-log", action="store_true", help="将 X 轴改为对数尺度（指数增长视图，基数10）")
    parser.add_argument("--debug", action="store_true", help="输出调试信息")
    args = parser.parse_args(list(argv) if argv is not None else None)

    paths = [Path(p) for p in args.files]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"文件不存在：{p}")
    grouped = group_runs_by_agent(paths)
    aggregated = aggregate_means_by_agent(grouped)
    if args.title is None:
        algos = sorted(grouped.keys())
        args.title = " vs. ".join(algos) if algos else "算法指标对比"
    _plot_2x2(
        aggregated,
        title=args.title,
        out_file=Path(args.out),
        min_cross_gap_ratio=args.min_cross_gap,
        show=bool(args.show),
        debug=bool(args.debug),
        x_log=bool(args.x_log),
    )
    print(f"✅ 图表已保存：{args.out}")
    return 0

if __name__ == "__main__":
    cli_main()
