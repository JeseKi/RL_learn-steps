"""UCB1算法模块：实现UCB1算法

公开接口：
- ucb1: UCB1算法实现
"""

import numpy as np

from .schemas import UCB1RewardsState


# UCB1 系算法
def ucb1(rewards: UCB1RewardsState, _: np.random.Generator, steps: int, **__) -> int:
    """UCB1 算法：基于置信区间的上界选择最优的老虎机"""
    if not rewards.ucb_states.ucb_inited:
        return rewards.ucb_states.ucb_inited_index

    log_steps = np.log(steps) if steps > 0 else 0.0
    counts_np = np.array(rewards.counts, dtype=np.float64)
    q_values = rewards.q_values

    ucb_values = q_values + np.sqrt(2 * log_steps / np.maximum(counts_np, 1))

    rewards.ucb_values = ucb_values

    return int(np.argmax(ucb_values))
