"""UCB1算法测试模块

测试 UCB1Algorithm 类的公开接口，包括初始化阶段、UCB计算、边界条件和错误路径。
"""

import pytest
import numpy as np

from core.environment import RLEnv
from ucb1.agent import UCBAgent, UCB1Algorithm
from ucb1.schemas import UCB1AlgorithmType


@pytest.fixture
def simple_env():
    """创建简单环境的 fixture，2个机器"""
    return RLEnv(machine_count=2, seed=42)


@pytest.fixture
def ucb1_algorithm():
    """创建 UCB1Algorithm 实例的 fixture"""
    return UCB1Algorithm(ucb1_type=UCB1AlgorithmType.UCB1)


@pytest.fixture
def ucb_agent(simple_env, ucb1_algorithm):
    """创建 UCBAgent 实例的 fixture"""
    return UCBAgent(name="test_ucb", env=simple_env, algorithm=ucb1_algorithm, seed=42)


def test_ucb1_initialization(ucb_agent):
    """测试初始化阶段：优先选择未探索过的臂"""
    # 初始所有 counts=0，应按顺序选择，第一次是 0
    ucb_agent.algorithm.ucb_init_state.ucb_inited_index = 0
    action = ucb_agent.algorithm.run()
    assert action == 0

    # 模拟拉动 arm 0
    ucb_agent.rewards.counts[0] = 1
    ucb_agent.rewards.values[0] = 1.0
    ucb_agent.rewards.q_values[0] = 1.0

    # 现在 counts[1]=0，应选择 index 1
    ucb_agent.algorithm.ucb_init_state.ucb_inited_index = 1
    action = ucb_agent.algorithm.run()
    assert action == 1


def test_ucb1_after_all_initialized(ucb_agent):
    """测试所有臂初始化后，使用 UCB 选择"""
    # 模拟初始化：每个臂拉动一次
    ucb_agent.rewards.counts = [1, 1]
    ucb_agent.rewards.values = [1.0, 0.5]
    ucb_agent.rewards.q_values = np.array([1.0, 0.5])
    ucb_agent.steps = 2
    ucb_agent.algorithm.ucb_init_state.ucb_inited = True  # 强制设置为已完成初始化

    action = ucb_agent.algorithm.run()

    # 预期：UCB0 = 1.0 + sqrt(2*ln(2)/1) ≈ 1.0 + 1.177 = 2.177
    #       UCB1 = 0.5 + sqrt(2*ln(2)/1) ≈ 0.5 + 1.177 = 1.677
    # 应选择臂 0
    assert action == 0


def test_ucb1_boundary_steps_zero(ucb_agent):
    """边界测试：steps=0 时，仍优先选择未探索臂"""
    ucb_agent.steps = 0
    ucb_agent.algorithm.ucb_init_state.ucb_inited_index = 0
    action = ucb_agent.algorithm.run()
    assert action == 0


def test_ucb1_all_counts_positive_low_steps(ucb_agent):
    """边界测试：所有 counts>0，但 steps 小，验证不报错"""
    ucb_agent.rewards.counts = [1, 1]
    ucb_agent.rewards.values = [0.5, 0.5]
    ucb_agent.rewards.q_values = np.array([0.5, 0.5])
    ucb_agent.steps = 1  # log(1) = 0
    ucb_agent.algorithm.ucb_init_state.ucb_inited = True

    action = ucb_agent.algorithm.run()
    # UCB = 0.5 + 0 = 0.5，选择任意（np.argmax 默认返回第一个最大值的索引）
    assert action == 0


def test_ucb1_error_path_division_avoided(ucb_agent):
    """错误路径测试：确保不会除零（由初始化逻辑避免）"""
    ucb_agent.rewards.counts = [1, 0]
    ucb_agent.steps = 2
    ucb_agent.algorithm.ucb_init_state.ucb_inited_index = 1

    # 应选择未探索的臂 1，不进入 UCB 计算
    action = ucb_agent.algorithm.run()
    assert action == 1


def test_ucb1_numpy_compatibility(ucb_agent):
    """测试 NumPy 兼容性：验证向量化计算"""
    ucb_agent.rewards.counts = [1, 1]
    ucb_agent.rewards.values = [1.0, 0.0]
    ucb_agent.rewards.q_values = np.array([1.0, 0.0])
    ucb_agent.steps = 10
    ucb_agent.algorithm.ucb_init_state.ucb_inited = True

    action = ucb_agent.algorithm.run()

    # log(10) ≈ 2.302
    # UCB0 ≈ 1.0 + sqrt(2*2.302/1) ≈ 1.0 + 2.145 = 3.145
    # UCB1 ≈ 0.0 + sqrt(2*2.302/1) ≈ 0.0 + 2.145 = 2.145
    # 应选择臂 0
    assert action == 0
