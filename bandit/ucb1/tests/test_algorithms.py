"""UCB1算法测试模块

测试 ucb1 函数的公开接口，包括初始化阶段、UCB计算、边界条件和错误路径。
"""

import pytest
import random
from core.environment import RLEnv
from ucb1.schemas import UCB1RewardsState
from ucb1.algorithms import ucb1


@pytest.fixture
def simple_env():
    """创建简单环境的 fixture，2个机器"""
    return RLEnv(machine_count=2, seed=42)


@pytest.fixture
def rewards_state(simple_env):
    """创建初始 RewardsState"""
    return UCB1RewardsState.from_env(simple_env)


@pytest.fixture
def rng():
    """随机数生成器 fixture"""
    return random.Random(42)


def test_ucb1_initialization(rewards_state, rng):
    """测试初始化阶段：所有 counts 为 0 时，选择第一个臂"""
    # 初始所有 counts=0，应选择 index 0
    action = ucb1(rewards_state, rng, 1)
    assert action == 0

    # 模拟拉动 arm 0
    rewards_state.counts[0] = 1
    rewards_state.values[0] = 1.0
    rewards_state.q_values[0] = 1.0

    # 现在 counts[1]=0，应选择 index 1
    action = ucb1(rewards_state, rng, 2)
    assert action == 1


def test_ucb1_after_all_initialized(simple_env, rng):
    """测试所有臂初始化后，使用 UCB 选择"""
    rewards = UCB1RewardsState.from_env(simple_env)
    # 模拟初始化：每个臂拉动一次
    rewards.counts = [1, 1]
    rewards.values = [1.0, 0.5]  # arm 0 奖励高
    rewards.q_values = [1.0, 0.5]

    # steps=2，所有初始化完成，计算 UCB
    action = ucb1(rewards, rng, 2)

    # 预期：UCB0 = 1 + sqrt(2*ln(2)/1) ≈ 1 + 1.177 = 2.177
    # UCB1 = 0.5 + 1.177 ≈ 1.677
    # 选择 0
    assert action == 0


def test_ucb1_boundary_steps_zero(rewards_state, rng):
    """边界测试：steps=0 时，仍优先选择未探索臂"""
    # steps=0，但 counts[0]=0，应选择 0
    action = ucb1(rewards_state, rng, 0)
    assert action == 0


def test_ucb1_all_counts_positive_low_steps(simple_env, rng):
    """边界测试：所有 counts>0，但 steps 小，验证不报错"""
    rewards = UCB1RewardsState.from_env(simple_env)
    rewards.counts = [1, 1]
    rewards.values = [0.5, 0.5]
    rewards.q_values = [0.5, 0.5]

    # steps=1，log(1)=0，UCB=0.5 + 0 =0.5，选择任意（假设 max index 取第一个）
    action = ucb1(rewards, rng, 1)
    assert action in [0, 1]  # 相等，应返回 index 0 (max 默认取第一个)


def test_ucb1_error_path_division_avoided(simple_env, rng):
    """错误路径测试：确保不会除零（由初始化逻辑避免）"""
    rewards = UCB1RewardsState.from_env(simple_env)
    # 故意设置一个 counts=0，但不应该进入 UCB 计算
    rewards.counts[1] = 0
    rewards.counts[0] = 1

    # steps=2，应选择未探索的 1，不计算 UCB
    action = ucb1(rewards, rng, 2)
    assert action == 1  # 未进入 UCB 计算，避免 /0
