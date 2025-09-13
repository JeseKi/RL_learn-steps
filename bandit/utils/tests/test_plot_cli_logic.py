import numpy as np

from utils.schemas import ProcessPoint, ProcessPointData, ProcessRun
from utils.plot import (
    aggregate_means_by_agent,
    find_pairwise_intersections_for_metric,
    find_axis_intersections_for_series,
)


def make_run(agent: str, steps, values: dict) -> ProcessRun:
    points = []
    for i, s in enumerate(steps):
        points.append(
            ProcessPoint(
                step=s,
                data=ProcessPointData(
                    agent_name=agent,
                    seed=42,
                    regret=float(values.get("regret", [0] * len(steps))[i]),
                    regret_rate=float(values.get("regret_rate", [0] * len(steps))[i]),
                    total_reward=float(values.get("total_reward", [0] * len(steps))[i]),
                    optimal_rate=float(values.get("optimal_rate", [0] * len(steps))[i]),
                    convergence_steps=0,
                ),
            )
        )
    return ProcessRun(points=points)


def test_aggregate_means_by_agent_basic():
    steps = list(range(1, 11))
    run1 = make_run("algoA", steps, {"total_reward": [float(s) for s in steps]})
    run2 = make_run("algoA", steps, {"total_reward": [float(2 * s) for s in steps]})
    grouped = {"algoA": [run1, run2]}
    agg = aggregate_means_by_agent(grouped)
    series_list = agg.metrics["total_reward"]
    assert len(series_list) == 1
    s = series_list[0]
    assert s.algorithm == "algoA"
    assert s.steps == steps
    np.testing.assert_allclose(s.values, [1.5 * x for x in steps], rtol=1e-6)


def test_find_pairwise_intersections_for_metric():
    steps = list(range(1, 11))
    y1 = [float(x) for x in steps]
    y2 = [float(11 - x) for x in steps]
    crosses = find_pairwise_intersections_for_metric(steps, {"A": y1, "B": y2}, min_gap_ratio=0.05)
    assert len(crosses) >= 1
    x = crosses[0].x
    assert abs(x - 5.5) < 0.51


def test_find_axis_intersections_for_series():
    steps = list(range(1, 11))
    # 使得后续值低于初始值，满足 X 轴标注规则
    y = [float(6 - x) for x in steps]
    marks = find_axis_intersections_for_series(steps, y)
    has_x_axis = any(m.axis == "x" for m in marks)
    assert has_x_axis
    has_y_axis = any(m.axis == "y" for m in marks)
    assert has_y_axis

