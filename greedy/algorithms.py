from typing import List
import random

from core import EpsilonDecreasingState, Rewards
# 当前奖励：最高回报的奖励（累计奖励）
def greedy_normal(rewards: Rewards, _rng=None, **_) -> int:
    """普通贪婪算法：选择当前累计奖励最高的老虎机"""
    return rewards.values.index(max(rewards.values))


def epsilon_greedy(rewards: Rewards, rng: random.Random, epsilon: float = 0.1, **_) -> int:
    """ε-贪婪算法：以 ε 的概率随机选择，否则选累计奖励最高的"""
    if rng.random() < epsilon:
        return rng.randint(0, len(rewards.values) - 1)
    else:
        return rewards.values.index(max(rewards.values))


def epsilon_decreasing_greedy(rewards: Rewards, rng: random.Random, epsilon_state: EpsilonDecreasingState, **_) -> int:
    """ε-递减贪婪算法：ε 随时间递减，其余时间选累计奖励最高的"""
    if rng.random() < epsilon_state.epsilon:
        action = rng.randint(0, len(rewards.values) - 1)
    else:
        action = rewards.values.index(max(rewards.values))

    epsilon_state.epsilon = max(
        epsilon_state.min_epsilon,
        epsilon_state.epsilon * epsilon_state.decay
    )
    return action

# 当前奖励：最高平均回报奖励
def greedy_average(rewards: Rewards, _rng=None, **_) -> int:
    """普通贪婪算法：选择当前平均奖励最高的老虎机"""
    average_rewards = [
        r / c if c > 0 else 0 for r, c in zip(rewards.values, rewards.counts)
    ]
    return average_rewards.index(max(average_rewards))


def epsilon_average(rewards: Rewards, rng: random.Random, epsilon: float = 0.1, **_) -> int:
    """ε-贪婪算法：基于平均奖励进行探索与利用"""
    if rng.random() < epsilon:
        return rng.randint(0, len(rewards.values) - 1)
    else:
        average_rewards = [
            r / c if c > 0 else 0 for r, c in zip(rewards.values, rewards.counts)
        ]
        return average_rewards.index(max(average_rewards))


def epsilon_decreasing_average(rewards: Rewards, rng: random.Random, epsilon_state: EpsilonDecreasingState, **_) -> int:
    """ε-递减贪婪算法：基于平均奖励，ε 随时间递减"""
    if rng.random() < epsilon_state.epsilon:
        action = rng.randint(0, len(rewards.values) - 1)
    else:
        average_rewards = [
            r / c if c > 0 else 0 for r, c in zip(rewards.values, rewards.counts)
        ]
        action = average_rewards.index(max(average_rewards))

    epsilon_state.epsilon = max(
        epsilon_state.min_epsilon,
        epsilon_state.epsilon * epsilon_state.decay
    )
    return action