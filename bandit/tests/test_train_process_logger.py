import json
from pathlib import Path

from core.environment import RLEnv
from greedy.agent import GreedyAgent, GreedyAlgorithm
from greedy.schemas import GreedyType
from utils.save_data import ProcessDataLogger
from utils.schemas import ProcessDataDump
from train import train


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
