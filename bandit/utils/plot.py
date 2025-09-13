"""
绘图工具模块：

公开接口：
- plot_metrics_history(agents, agent_name, file_name): 从 Agent 的 metrics_history 绘制平均曲线
- plot_process_json(json_path, metric_key, title, file_name): 从过程数据 JSON 绘制指定指标

数据流预期：
- 训练过程若使用 ProcessDataLogger 进行采样与记录，则 metrics_history 中仅在网格步采样；
- plot_metrics_history 直接基于各 Agent 的 metrics_history 聚合后绘图；
- 若已将过程数据保存为 JSON，可使用 plot_process_json 读取并绘制单条指标随步数变化曲线。
"""

from typing import Any, List, Dict, Set

from core import BaseAgent

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from pathlib import Path


def plot_metrics_history(agents: List[BaseAgent], agent_name: str, file_name: Path):
    """
    根据训练后的一组 agent 的 metrics_history 绘制指标变化图。

    Args:
        agents (List[GreedyAgent]): 经过训练的 agent 列表。
        agent_name (str): 这组 agent 的名称，用于图表标题。
        file_name (Path): 保存的文件名（例如："experiment_result.png"）
    """

    if not agents:
        raise ValueError("Agents 列表为空，无法绘图")

    # 1. 设置字体
    font_path = Path.cwd() / "assets" / "微软雅黑.ttf"
    if font_path.exists():
        font_prop = FontProperties(fname=font_path, size=12)
        title_font_prop = FontProperties(fname=font_path, size=16)
    else:
        print(f"警告：找不到字体文件 {font_path}，将使用默认字体，中文可能显示为方框。")
        font_prop = FontProperties(size=12)
        title_font_prop = FontProperties(size=16)

    # 2. 准备数据

    # 收集所有记录的时间步
    recorded_steps: Set[int] = set()
    for agent in agents:
        for _, _, step in agent.metrics_history:
            recorded_steps.add(step)

    # 转换为排序后的列表
    recorded_steps_list: List[int] = sorted(list(recorded_steps))

    # 初始化存储指标历史的字典
    metrics_history: Dict[str, Any] = {
        "regret": [],
        "regret_rate": [],
        "total_reward": [],
        "optimal_rate": [],
        "convergence_steps": [],
        "convergence_rate": [],
    }

    # 遍历所有记录的时间步
    for step in recorded_steps_list:
        # 临时存储当前时间步所有 agent 的指标
        step_metrics: Dict[str, Any] = {
            "regret": [],
            "regret_rate": [],
            "total_reward": [],
            "optimal_rate": [],
            "convergence_steps": [],
            "convergence_rate": [],
        }

        # 遍历每个 agent，查找在当前时间步记录的数据
        for agent in agents:
            # 在 agent 的 metrics_history 中查找对应时间步的数据
            for rewards_state, metrics, recorded_step in agent.metrics_history:
                if recorded_step == step:
                    step_metrics["regret"].append(metrics.regret)
                    step_metrics["regret_rate"].append(metrics.regret_rate)
                    step_metrics["total_reward"].append(sum(metrics.rewards.values))
                    step_metrics["optimal_rate"].append(metrics.optimal_rate)
                    step_metrics["convergence_steps"].append(agent.convergence_steps)
                    step_metrics["convergence_rate"].append(
                        1 if agent.convergence_steps > 0 else 0
                    )
                    break  # 找到对应时间步的数据后跳出循环

        # 计算当前时间步的平均值并存入 metrics_history
        for key in metrics_history:
            if step_metrics[key]:  # 确保列表不为空
                metrics_history[key].append(np.mean(step_metrics[key]))
            else:
                # 如果没有 agent 在这个时间步有数据，添加 NaN
                metrics_history[key].append(np.nan)

    # 3. 开始绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
    fig.suptitle(f'"{agent_name}" 算法平均指标变化情况', fontproperties=title_font_prop)

    assert isinstance(axes, np.ndarray)
    axes = axes.flatten()

    plot_config = {
        "regret": "后悔值 (Regret)",
        "regret_rate": "后悔率 (Regret Rate)",
        "total_reward": "累积总奖励 (Total Reward)",
        "optimal_rate": "最优臂选择率 (Optimal Rate)",
        "convergence_steps": "达到收敛时的步数 (Convergence Steps)",
        "convergence_rate": "达到收敛率 (Convergence Rate)",
    }

    for i, (metric_key, title) in enumerate(plot_config.items()):
        ax = axes[i]
        # 只在有数据记录的时间步绘制数据点
        ax.plot(
            recorded_steps_list,
            metrics_history[metric_key],
            label=title,
            marker="o",
            markersize=3,
        )
        ax.set_title(title, fontproperties=font_prop)
        ax.set_xlabel("时间步 (Steps)", fontproperties=font_prop)
        ax.set_ylabel("平均值", fontproperties=font_prop)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(prop=font_prop)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()

    if not file_name.suffix == ".png":
        file_name = file_name.with_suffix(".png")
    fig.savefig(file_name)
    print(f"✅ 图表已保存至 {file_name}")
