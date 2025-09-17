"""Core 模块 environment.py 测试

文件功能：
- 针对 `core.environment.SlotMachine` 和 `core.environment.RLEnv` 的公开接口进行单元测试，覆盖常规、边界和错误路径。

公开接口：
- pytest 测试函数：
  - `test_slot_machine_pull_returns_valid_reward`
  - `test_rl_env_pull_returns_valid_reward`
  - `test_rl_env_pull_invalid_machine_id_raises`
  - `test_rl_env_best_reward`
  - `test_rl_env_random_walk_reward`
  - `test_rl_env_piecewise_reward_upside_down`
  - `test_rl_env_piecewise_reward_reset`
  - `test_rl_env_piecewise_reward_addition_subtraction`

内部方法：
- 本测试文件不包含内部辅助方法。

公开接口的 pydantic 模型：
- 不适用（测试文件无对外返回模型）。
"""

from __future__ import annotations

import pytest

# 使用相对导入
from ..environment import SlotMachine, RLEnv
from ..schemas import PiecewizeMethod


def test_slot_machine_pull_returns_valid_reward():
    """常规路径：测试老虎机拉动返回有效的奖励值（0或1）"""
    machine = SlotMachine(reward_probability=0.5, seed=42)
    
    # 拉动老虎机多次，检查返回值是否为0或1
    for _ in range(100):
        reward = machine.pull()
        assert reward in [0, 1]


def test_rl_env_pull_returns_valid_reward():
    """常规路径：测试环境拉动机器返回有效的奖励值（0或1）"""
    env = RLEnv(machine_count=5, seed=42)
    
    # 拉动每个机器多次，检查返回值是否为0或1
    for machine_id in range(5):
        for _ in range(20):
            reward = env.pull(machine_id, steps=0)
            assert reward in [0, 1]


def test_rl_env_pull_invalid_machine_id_raises():
    """错误路径：非法机器 ID 应触发值异常"""
    env = RLEnv(machine_count=3, seed=42)
    
    # 测试超出上界的机器ID
    with pytest.raises(ValueError, match="机器ID超出范围"):
        _ = env.pull(3, steps=0)  # 有效索引为 [0, 2]
    
    # 测试负数机器ID
    with pytest.raises(ValueError, match="机器ID超出范围"):
        _ = env.pull(-1, steps=0)


def test_rl_env_best_reward():
    """常规路径：测试环境计算最佳奖励的方法"""
    env = RLEnv(machine_count=5, seed=42)
    
    # 获取最佳机器的奖励概率
    best_prob = env.best_reward_machine.reward_probability
    
    # 测试不同步数下的最佳奖励计算
    steps = 100
    expected_best_reward = best_prob * steps
    assert env.best_reward(steps) == expected_best_reward


def test_rl_env_random_walk_reward():
    """常规路径：测试环境的随机游走奖励机制"""
    # 创建具有随机游走特性的环境
    env = RLEnv(
        machine_count=5, 
        random_walk_internal=10,  # 每10步进行一次随机游走
        random_walk_machine_num=2,  # 每次选择2个机器进行随机游走
        seed=42
    )
    
    # 保存原始奖励概率
    original_probs = [machine.reward_probability for machine in env.machines]
    
    # 执行一次随机游走（steps=10，触发随机游走）
    _ = env.pull(0, steps=10)
    
    # 检查是否有机器的奖励概率发生了变化
    new_probs = [machine.reward_probability for machine in env.machines]
    
    # 至少有一个机器的概率应该发生了变化
    assert any(orig != new for orig, new in zip(original_probs, new_probs))


def test_rl_env_piecewise_reward_upside_down():
    """常规路径：测试环境的分段奖励机制（颠倒）"""
    env = RLEnv(
        machine_count=5,
        piecewise_internal=5,  # 每5步进行一次分段变化
        piecewize_method=PiecewizeMethod.UPSIDE_DOWN,
        seed=42
    )
    
    # 执行分段变化（steps=5，触发分段变化）
    _ = env.pull(0, steps=5)
    
    # 获取新的奖励概率并排序
    new_probs = sorted([machine.reward_probability for machine in env.machines])
    
    # 验证是否正确实现了颠倒逻辑
    n = len(new_probs)
    expected_probs = sorted([(n - 1 - i) / n for i in range(n)])
    
    # 检查新的概率是否符合预期
    for new, expected in zip(new_probs, expected_probs):
        assert abs(new - expected) < 1e-10


def test_rl_env_piecewise_reward_reset():
    """常规路径：测试环境的分段奖励机制（重置）"""
    env = RLEnv(
        machine_count=5,
        piecewise_internal=5,  # 每5步进行一次分段变化
        piecewize_method=PiecewizeMethod.RESET,
        seed=42
    )
    
    # 保存初始奖励概率
    initial_probs = [machine.reward_probability for machine in env.machines]
    
    # 修改当前奖励概率
    for machine in env.machines:
        machine.reward_probability = 0.5
    
    # 执行分段变化（steps=5，触发分段变化）
    _ = env.pull(0, steps=5)
    
    # 获取重置后的奖励概率
    reset_probs = [machine.reward_probability for machine in env.machines]
    
    # 检查是否恢复到初始状态（忽略顺序，因为 _reset 会 shuffle）
    assert sorted(reset_probs) == sorted(initial_probs)


def test_rl_env_piecewise_reward_addition_subtraction():
    """常规路径：测试环境的分段奖励机制（加减）"""
    env = RLEnv(
        machine_count=5,
        piecewise_internal=5,  # 每5步进行一次分段变化
        piecewize_method=PiecewizeMethod.ADDITION_SUBTRACTION,
        seed=42
    )
    
    # 保存原始奖励概率
    original_probs = [machine.reward_probability for machine in env.machines]
    
    # 执行分段变化（steps=5，触发分段变化）
    _ = env.pull(0, steps=5)
    
    # 获取新的奖励概率
    new_probs = [machine.reward_probability for machine in env.machines]
    
    # 验证加减逻辑：每个概率应该增加0.5，但如果超过1则减少0.5
    for orig, new in zip(original_probs, new_probs):
        expected = orig + 0.5
        if expected > 1:
            expected = orig - 0.5
        assert abs(new - expected) < 1e-10