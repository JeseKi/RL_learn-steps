import numpy as np

from utils.schemas import ProcessPoint, ProcessPointData, ProcessRun
from utils.plot import (
    aggregate_means_by_agent,
    find_pairwise_intersections_for_metric,
    find_axis_intersections_for_series,
)
from utils.plot_cli import compute_avg_convergence_steps_by_algo


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


def make_run_with_duplicates(agent: str, steps, values_a: dict, values_b: dict) -> ProcessRun:
    """构造在同一步存在多条记录（模拟多 seed）的 ProcessRun。

    - values_a/values_b: 按指标提供两组值，长度与 steps 一致；
    - 本构造函数会为每个 step 追加两条记录（seed=1 与 seed=2）。
    """
    points = []
    for i, s in enumerate(steps):
        # 第一条记录（seed=1）
        points.append(
            ProcessPoint(
                step=s,
                data=ProcessPointData(
                    agent_name=agent,
                    seed=1,
                    regret=float(values_a.get("regret", [0] * len(steps))[i]),
                    regret_rate=float(values_a.get("regret_rate", [0] * len(steps))[i]),
                    total_reward=float(values_a.get("total_reward", [0] * len(steps))[i]),
                    optimal_rate=float(values_a.get("optimal_rate", [0] * len(steps))[i]),
                    convergence_steps=0,
                ),
            )
        )
        # 第二条记录（seed=2）
        points.append(
            ProcessPoint(
                step=s,
                data=ProcessPointData(
                    agent_name=agent,
                    seed=2,
                    regret=float(values_b.get("regret", [0] * len(steps))[i]),
                    regret_rate=float(values_b.get("regret_rate", [0] * len(steps))[i]),
                    total_reward=float(values_b.get("total_reward", [0] * len(steps))[i]),
                    optimal_rate=float(values_b.get("optimal_rate", [0] * len(steps))[i]),
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


def test_aggregate_means_handles_duplicates_within_run():
    # 构造单个 run，其中每个 step 有两条记录（模拟两个 seed），
    # total_reward 两组分别是 s 与 2*s，聚合后应为 1.5*s。
    steps = [1, 2, 3, 4]
    run_dup = make_run_with_duplicates(
        "algoX",
        steps,
        {"total_reward": [float(s) for s in steps]},
        {"total_reward": [float(2 * s) for s in steps]},
    )
    grouped = {"algoX": [run_dup]}
    agg = aggregate_means_by_agent(grouped)
    series_list = agg.metrics["total_reward"]
    assert len(series_list) == 1
    s = series_list[0]
    assert s.algorithm == "algoX"
    assert s.steps == steps
    import numpy as np
    np.testing.assert_allclose(s.values, [1.5 * x for x in steps], rtol=1e-6)


def test_compute_avg_convergence_steps_by_algo():
    # 构造包含两个 seed 的 run：seed1 收敛步数 1000，seed2 收敛步数 2000
    steps = [10, 100000]
    points = []
    # seed 1：在最后一步写入 convergence_steps=1000
    points.append(
        ProcessPoint(
            step=10,
            data=ProcessPointData(
                agent_name="algoC",
                seed=1,
                regret=0.0,
                regret_rate=0.0,
                total_reward=0.0,
                optimal_rate=0.0,
                convergence_steps=0,
            ),
        )
    )
    points.append(
        ProcessPoint(
            step=100000,
            data=ProcessPointData(
                agent_name="algoC",
                seed=1,
                regret=0.0,
                regret_rate=0.0,
                total_reward=0.0,
                optimal_rate=1.0,
                convergence_steps=1000,
            ),
        )
    )
    # seed 2：在最后一步写入 convergence_steps=2000
    points.append(
        ProcessPoint(
            step=10,
            data=ProcessPointData(
                agent_name="algoC",
                seed=2,
                regret=0.0,
                regret_rate=0.0,
                total_reward=0.0,
                optimal_rate=0.0,
                convergence_steps=0,
            ),
        )
    )
    points.append(
        ProcessPoint(
            step=100000,
            data=ProcessPointData(
                agent_name="algoC",
                seed=2,
                regret=0.0,
                regret_rate=0.0,
                total_reward=0.0,
                optimal_rate=1.0,
                convergence_steps=2000,
            ),
        )
    )
    run = ProcessRun(points=points)

    grouped = {"algoC": [run]}
    avg = compute_avg_convergence_steps_by_algo(grouped)
    assert "algoC" in avg
    assert abs(avg["algoC"] - 1500.0) < 1e-6
