import json
from pathlib import Path

from core.environment import RLEnv
from greedy.agent import GreedyAgent, GreedyAlgorithm
from greedy.schemas import GreedyType
from utils.save_data import ProcessDataLogger
from utils.schemas import ProcessDataDump
from train import train, batch_train


def _parallel_agent_factory(
    env: RLEnv,
    seed: int,
    convergence_threshold: float,
    convergence_min_steps: int,
) -> GreedyAgent:
    """batch_train 并行测试专用的 Agent 工厂函数"""

    algorithm = GreedyAlgorithm(greedy_type=GreedyType.GREEDY)
    return GreedyAgent(
        name="greedy_parallel",
        env=env,
        algorithm=algorithm,
        seed=seed,
        convergence_threshold=convergence_threshold,
        convergence_min_steps=convergence_min_steps,
    )


def test_train_with_process_logger(tmp_path: Path):
    # 构造环境与单个 Agent
    env = RLEnv(machine_count=5, seed=123)
    algorithm = GreedyAlgorithm(greedy_type=GreedyType.GREEDY)
    agent = GreedyAgent(
        name="greedy_avg",
        env=env,
        algorithm=algorithm,
        seed=1234,
    )

    steps = 120
    logger = ProcessDataLogger(run_id="case-train", total_steps=steps, grid_size=20)

    agents, avg_rewards, avg_metrics = train(
        agents=[agent],
        steps=steps,
        process_logger=logger,
    )

    # metrics_history 的记录次数应与 grid 一致
    hist_steps = [s for (_, _, s) in agents[0].metrics_history]
    assert hist_steps == logger.grid

    # 保存并校验 JSON 结构
    out = tmp_path / "proc.json"
    logger.save(out, total_steps=steps)
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    dump = ProcessDataDump(**data)

    recorded_steps = [p.step for p in dump.points]
    assert recorded_steps == logger.grid
    # 基本字段存在
    assert all("total_reward" in p.data for p in dump.points)


def test_batch_train_parallel_generates_independent_runs():
    env = RLEnv(machine_count=5, seed=321)
    steps = 120
    logger = ProcessDataLogger(run_id="case-batch", total_steps=steps, grid_size=20)

    agents, _, _ = batch_train(
        count=3,
        agent_factory=_parallel_agent_factory,
        env=env,
        steps=steps,
        seed=100,
        convergence_threshold=0.9,
        convergence_min_steps=50,
        process_logger=logger,
        num_workers=3,
    )

    assert len(agents) == 3

    dump = logger.export(total_steps=steps)
    by_seed = {}
    for point in dump.points:
        agent_seed = point.data["seed"]
        by_seed.setdefault(agent_seed, []).append((point.step, point.data["total_reward"]))

    assert len(by_seed) == 3

    sequences = {
        tuple(value for _, value in sorted(seq, key=lambda item: item[0]))
        for seq in by_seed.values()
    }

    # 不同种子应得到不同的奖励轨迹，从侧面验证环境随机流已经隔离
    assert len(sequences) == len(by_seed)
