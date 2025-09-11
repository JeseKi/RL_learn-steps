from pathlib import Path
import json

from core.schemas import BaseRewardsState
from train import AverageMetrics


def save_experiment_data(
    reward: BaseRewardsState,
    metrics: AverageMetrics,
    file_name: Path
):
    """
    将实验结果保存为 JSON 文件。
    
    Args:
        agents: 训练后的所有 agent 实例列表
        reward: 平均奖励信息
        metrics: 平均评估指标
        file_name: 保存的文件名（例如："experiment_result.json"）
    """
    
    # 转换 Agents 数据为可序列化的字典
    experiment_data = {
        "reward": reward.model_dump(),
        "metrics": metrics.model_dump(),
    }

    # 写入文件
    if not file_name.parent.exists():
        file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 实验数据已保存至 {file_name}")