import json
from pathlib import Path

from core.environment import RLEnv
from greedy.agent import GreedyAgent
from greedy.algorithms import greedy_average
from utils.save_data import ProcessDataLogger
from utils.schemas import ProcessDataDump
from train import train


def test_greedy_optimistic_init_steps_reach_T_and_log_grid(tmp_path: Path):
    """当启用乐观初始化时，也应当在每次 act 后累计全局步数，
    确保训练结束后 agent.steps == T，且过程记录命中网格中的最后一步 T。
    """
    env = RLEnv(machine_count=5, seed=2024)
    agent = GreedyAgent(
        name="greedy_avg_opt",
        env=env,
        greedy_algorithm=greedy_average,
        optimistic_init=True,
        optimistic_times=1,
        seed=42,
    )

    steps = 200
    logger = ProcessDataLogger(run_id="case-opt", total_steps=steps, grid_size=30)

    agents, avg_rewards, avg_metrics = train(
        agents=[agent],
        steps=steps,
        process_logger=logger,
    )

    # 训练结束后，代理的全局步数应正好为 T
    assert agents[0].steps == steps

    # 保存并检查记录的步数应与网格一致，且包含末尾的 T
    out = tmp_path / "proc.json"
    logger.save(out, total_steps=steps)
    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)
    dump = ProcessDataDump(**data)

    recorded_steps = [p.step for p in dump.points]
    assert recorded_steps == logger.grid
    assert recorded_steps[-1] == steps

