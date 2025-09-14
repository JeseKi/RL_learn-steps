"""
文件功能：
- 基于 Agent.metrics_history 的快速绘图（兼容保留）。

公开接口：
- plot_metrics_history(agents, agent_name, file_name)

内部方法：
- 无。

公开接口的 pydantic 模型：
- 无（直接消费 BaseAgent.metrics_history）。
"""

from typing import Any, Dict, List, Set
from pathlib import Path
import numpy as np

from core import BaseAgent
from utils.plot_font import _ensure_matplotlib


def plot_metrics_history(
    agents: List[BaseAgent], agent_name: str, file_name: Path, x_log: bool = False
):
    """根据训练后的一组 agent 的 metrics_history 绘制指标变化图。

    Args:
        agents: 训练后的 agent 列表
        agent_name: agent 名称
        file_name: 保存文件路径
        x_log: 是否使用对数刻度显示 X 轴
    """
    if not agents:
        raise ValueError("Agents 列表为空，无法绘图")

    # 1. 设置字体/绘图库
    plt, font_prop, title_font_prop = _ensure_matplotlib()

    # 收集所有记录的时间步
    recorded_steps: Set[int] = set()
    for agent in agents:
        for _, _, step in agent.metrics_history:
            recorded_steps.add(step)
    recorded_steps_list: List[int] = sorted(list(recorded_steps))

    metrics_history: Dict[str, Any] = {
        "regret": [],
        "regret_rate": [],
        "total_reward": [],
        "optimal_rate": [],
        "convergence_steps": [],
        "convergence_rate": [],
    }

    for step in recorded_steps_list:
        step_metrics: Dict[str, Any] = {
            "regret": [],
            "regret_rate": [],
            "total_reward": [],
            "optimal_rate": [],
            "convergence_steps": [],
            "convergence_rate": [],
        }
        for agent in agents:
            for _, metrics, recorded_step in agent.metrics_history:
                if recorded_step == step:
                    step_metrics["regret"].append(metrics.regret)
                    step_metrics["regret_rate"].append(metrics.regret_rate)
                    step_metrics["total_reward"].append(sum(metrics.rewards.values))
                    step_metrics["optimal_rate"].append(metrics.optimal_rate)
                    step_metrics["convergence_steps"].append(agent.convergence_steps)
                    step_metrics["convergence_rate"].append(
                        1 if agent.convergence_steps > 0 else 0
                    )
                    break
        for key in metrics_history:
            metrics_history[key].append(
                float(np.mean(step_metrics[key])) if step_metrics[key] else float("nan")
            )

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)
    fig.suptitle(f'"{agent_name}" 算法平均指标变化情况', fontproperties=title_font_prop)
    axes = np.array(axes).flatten()

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
        if x_log:
            ax.set_xscale("log")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(prop=font_prop)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    file_name = (
        file_name.with_suffix(".png") if file_name.suffix != ".png" else file_name
    )
    file_name = file_name.with_stem(file_name.stem + "_x_log") if x_log else file_name
    fig.savefig(file_name)
    print(f"✅ 图表已保存至 {file_name}")
