"""UCB1算法模块：实现UCB1算法

公开接口：
- ucb1: UCB1算法实现
"""

import random
import math

from .schemas import UCB1RewardsState


# UCB1 系算法
def ucb1(rewards: UCB1RewardsState, _: random.Random, steps: int, **__) -> int:
    """UCB1 算法：基于置信区间的上界选择最优的老虎机"""
    for i in range(len(rewards.values)):  # 更新所有机器的UCB值
        if not rewards.ucb_states.ucb_inited:
            return rewards.ucb_states.ucb_inited_index
        rewards.ucb_values[i] = rewards.q_values[i] + math.sqrt(
            2 * math.log(steps) / rewards.counts[i]
        )
    return rewards.ucb_values.index(max(rewards.ucb_values))
