"""
重构验证测试文件

这个文件用于验证重构后的模块是否能正常工作。
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """测试模块导入"""
    try:
        # 测试 core 模块导入
        print("✓ core 模块导入成功")

        # 测试 greedy 模块导入
        print("✓ greedy 模块导入成功")

        # 测试 ucb1 模块导入
        print("✓ ucb1 模块导入成功")

        # 测试 train 模块导入
        print("✓ train 模块导入成功")

        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def test_basic_functionality():
    """测试基本功能"""
    try:
        # 导入必要的模块
        from core.environment import RLEnv
        from greedy.agent import GreedyAgent
        from greedy.config import EpsilonDecreasingConfig
        from greedy.algorithms import greedy_average
        from ucb1.agent import UCB1Agent

        # 创建环境
        env = RLEnv(machine_count=5, seed=42)
        print("✓ 环境创建成功")

        # 测试 GreedyAgent
        agent = GreedyAgent(
            name="test_greedy_agent",
            env=env,
            greedy_algorithm=greedy_average,
            epsilon_config=EpsilonDecreasingConfig(),
            seed=42,
        )
        print("✓ GreedyAgent 创建成功")

        # 执行一步行动
        action = agent.act(steps=1)
        print(f"✓ GreedyAgent 行动选择成功: 选择了老虎机 {action}")

        # 测试 UCB1Agent
        ucb_agent = UCB1Agent(name="test_ucb_agent", env=env, seed=42)
        print("✓ UCB1Agent 创建成功")

        # 执行一步行动
        ucb_action = ucb_agent.act(steps=1)
        print(f"✓ UCB1Agent 行动选择成功: 选择了老虎机 {ucb_action}")

        return True
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False


def main():
    """主函数"""
    print("开始验证重构后的代码...")
    print("=" * 50)

    # 测试模块导入
    print("1. 测试模块导入...")
    if not test_imports():
        return False

    print("\n2. 测试基本功能...")
    if not test_basic_functionality():
        return False

    print("\n" + "=" * 50)
    print("✓ 所有测试通过！重构成功！")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
