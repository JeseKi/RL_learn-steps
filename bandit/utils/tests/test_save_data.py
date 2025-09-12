import json
from pathlib import Path


from utils.save_data import ProcessDataLogger
from utils.schemas import ProcessDataDump


def test_should_record_grid():
    total_steps = 250
    grid_size = 50
    logger = ProcessDataLogger(run_id="t1", total_steps=total_steps, grid_size=grid_size)

    # 网格应包含 1 与 T，并严格递增且不越界
    grid = logger.grid
    assert len(grid) > 0
    assert grid[0] >= 1 and grid[-1] == total_steps
    assert all(1 <= x <= total_steps for x in grid)
    assert all(grid[i] < grid[i + 1] for i in range(len(grid) - 1))

    # should_record 为 True 的次数应与网格长度一致
    true_count = sum(1 for s in range(1, total_steps + 1) if logger.should_record(s))
    assert true_count == len(grid)


def test_add_and_save_process_data(tmp_path: Path):
    total_steps = 250
    grid_size = 50
    logger = ProcessDataLogger(run_id="run-xyz", total_steps=total_steps, grid_size=grid_size)
    for s in range(1, total_steps + 1):
        if logger.should_record(s):
            logger.add(s, {"reward": float(s), "note": f"第{s}步"})

    out_file = tmp_path / "process.json"
    path = logger.save(out_file, total_steps=total_steps)
    assert path.exists(), "输出文件应存在"
    assert path.suffix == ".json"

    # 读取并验证结构
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dump = ProcessDataDump(**data)
    assert dump.run_id == "run-xyz"
    assert dump.total_steps == total_steps
    assert len(dump.points) > 0
    # 记录的步集合应与网格一致
    steps = [p.step for p in dump.points]
    assert steps == logger.grid
