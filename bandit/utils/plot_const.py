"""
文件功能：
- 常量与类型别名定义，服务于绘图与聚合模块。

公开接口：
- METRICS_FOR_2x2：2x2 图表需要绘制的四个指标键名元组。

内部方法：
- 无。

公开接口的 pydantic 模型：
- 无（仅常量）。
"""

from typing import Tuple


# 2x2 子图的指标顺序（外部模块共同依赖）
METRICS_FOR_2x2: Tuple[str, str, str, str] = (
    "total_reward",
    "optimal_rate",
    "regret",
    "regret_rate",
)
