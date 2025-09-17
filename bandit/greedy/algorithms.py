"""贪婪算法模块：实现各种贪婪算法

公开接口：
- greedy_normal: 普通贪婪算法（基于累计奖励）
- epsilon_greedy: ε-贪婪算法（基于累计奖励）
- epsilon_decreasing_greedy: ε-递减贪婪算法（基于累计奖励）
- greedy_average: 普通贪婪算法（基于平均奖励）
- epsilon_average: ε-贪婪算法（基于平均奖励）
- epsilon_decreasing_average: ε-递减贪婪算法（基于平均奖励）
"""

import numpy as np

from .schemas import GreedyRewardsState
from .config import EpsilonDecreasingState


# 当前奖励：最高回报的奖励（累计奖励）
def greedy_normal(rewards: GreedyRewardsState, _=None, **__) -> int:
    """普通贪婪算法：选择当前累计奖励最高的老虎机"""
    return rewards.values.index(max(rewards.values))


def epsilon_greedy(
    rewards: GreedyRewardsState, rng: np.random.Generator, epsilon: float = 0.1, **_
) -> int:
    """ε-贪婪算法：以 ε 的概率随机选择，否则选累计奖励最高的"""
    if rng.random() < epsilon:
        return int(rng.integers(0, len(rewards.values)))
    else:
        return rewards.values.index(max(rewards.values))


def epsilon_decreasing_greedy(
    rewards: GreedyRewardsState,
    rng: np.random.Generator,  # 已改为 np.random.Generator
    epsilon_state: EpsilonDecreasingState,
    **_,
) -> int:
    """ε-递减贪婪算法：ε 随时间递减，其余时间选累计奖励最高的"""
    if rng.random() < epsilon_state.epsilon:
        action = int(rng.integers(0, len(rewards.values)))
    else:
        action = rewards.values.index(max(rewards.values))

    epsilon_state.epsilon = max(
        epsilon_state.min_epsilon, epsilon_state.epsilon * epsilon_state.decay
    )
    return int(action)


# 当前奖励：最高平均回报奖励
def greedy_average(rewards: GreedyRewardsState, _rng=None, **_) -> int:
    """普通贪婪算法：选择当前 Q 值最高的老虎机"""
    return rewards.q_values.index(max(rewards.q_values))


def epsilon_average(
    rewards: GreedyRewardsState, rng: np.random.Generator, epsilon: float = 0.1, **_
) -> int:
    """ε-贪婪算法：基于 Q 值进行探索与利用"""
    if rng.random() < epsilon:
        return int(rng.integers(0, len(rewards.q_values)))
    else:
        return rewards.q_values.index(max(rewards.q_values))


def epsilon_decreasing_average(
    rewards: GreedyRewardsState,
    rng: np.random.Generator,  # 已改为 np.random.Generator
    epsilon_state: EpsilonDecreasingState,
    **_,
) -> int:
    """ε-递减贪婪算法：基于 Q 值，ε 随时间递减"""
    if rng.random() < epsilon_state.epsilon:
        action = int(rng.integers(0, len(rewards.q_values)))
    else:
        action = rewards.q_values.index(max(rewards.q_values))

    epsilon_state.epsilon = max(
        epsilon_state.min_epsilon, epsilon_state.epsilon * epsilon_state.decay
    )
    return int(action)
