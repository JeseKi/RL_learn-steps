"""
多臂老虎机算法使用示例

这个文件展示了如何使用重构后的模块来运行多臂老虎机实验。
"""

from typing import Callable, List, Tuple, Optional
import gc

from core import BaseRewardsState, RLEnv
from core.agent import BaseAgent
from greedy import (
    EpsilonDecreasingConfig,
    GreedyAgent,
    greedy_average,
    epsilon_average,
    epsilon_decreasing_average,
)
from ucb1 import UCB1Agent
from train import train, AverageMetrics


# 实验配置参数
SEED: int = 42
MACHINE_COUNT: int = 100
COUNT: int = 50
STEPS: int = 10_000
CONVERGENCE_THRESHOLD: float = 0.9
CONVERGENCE_MIN_STEPS: int = 1000
OPTIMISTIC_TIMES: int = 1
ENABLE_OPTIMISTIC: bool = True


def batch_train(
    count: int,
    agent_factory: Callable[..., BaseAgent],
    env: RLEnv,
    epsilon_config: Optional[EpsilonDecreasingConfig] = None,
    steps: int = STEPS,
    convergence_threshold: float = CONVERGENCE_THRESHOLD,
    convergence_min_steps: int = CONVERGENCE_MIN_STEPS,
    seed: int = SEED,
    **kwargs,
) -> Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]:
    """批训练 Agent，传入数量，代理工厂函数，环境，步数和初始种子即可训练

    Args:
        count (int): 训练数量
        agent_factory (Callable[..., BaseAgent]): 创建代理的工厂函数
        env (RLEnv): 环境
        epsilon_config (Optional[EpsilonDecreasingConfig], optional): ε-递减配置
        steps (int): 步数
        convergence_threshold (float): 收敛阈值
        convergence_min_steps (int): 最小收敛步数
        seed (int): 初始种子
        **kwargs: 传递给代理工厂函数的额外参数

    Returns:
        Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]: 训练结果
    """
    _agents: List[BaseAgent] = []

    for i in range(count):
        agent_kwargs = {
            "seed": seed + i,
            "convergence_threshold": convergence_threshold,
            "convergence_min_steps": convergence_min_steps,
            **kwargs,
        }
        if epsilon_config is not None:
            agent_kwargs["epsilon_config"] = epsilon_config
        _agents.append(agent_factory(env, **agent_kwargs))

    agents, reward, metrics = train(_agents, steps)
    return agents, reward, metrics


def run_experiment():
    """运行多臂老虎机实验"""
    # 创建环境
    env: RLEnv = RLEnv(machine_count=MACHINE_COUNT, seed=SEED)

    # 创建ε-递减配置
    epsilon_config = EpsilonDecreasingConfig()

    print("开始运行多臂老虎机实验...")
    print(f"老虎机数量: {MACHINE_COUNT}")
    print(f"训练次数: {COUNT}")
    print(f"步数: {STEPS}")
    print("-" * 50)

    # 1. 普通贪婪算法
    def create_greedy_agent(env: RLEnv, **kwargs) -> BaseAgent:
        return GreedyAgent(
            name="greedy_average",
            env=env,
            greedy_algorithm=greedy_average,
            optimistic_init=ENABLE_OPTIMISTIC,
            optimistic_times=OPTIMISTIC_TIMES,
            **kwargs,
        )

    print("1. 运行普通贪婪算法...")
    agents, reward, metrics = batch_train(
        count=COUNT,
        agent_factory=create_greedy_agent,
        env=env,
        epsilon_config=epsilon_config,
        steps=STEPS,
        seed=SEED,
    )

    print(f"算法名称: {agents[0].name}")
    print(f"平均奖励: {reward}")
    print(f"指标: {metrics}")
    print("-" * 50)

    # 清理内存
    del agents, reward, metrics
    gc.collect()

    # 2. ε-贪婪算法
    def create_epsilon_agent(env: RLEnv, **kwargs) -> BaseAgent:
        return GreedyAgent(
            name="epsilon_average",
            env=env,
            greedy_algorithm=epsilon_average,
            optimistic_init=ENABLE_OPTIMISTIC,
            optimistic_times=OPTIMISTIC_TIMES,
            **kwargs,
        )

    print("2. 运行ε-贪婪算法...")
    agents, reward, metrics = batch_train(
        count=COUNT,
        agent_factory=create_epsilon_agent,
        env=env,
        epsilon_config=epsilon_config,
        steps=STEPS,
        seed=SEED,
    )

    print(f"算法名称: {agents[0].name}")
    print(f"平均奖励: {reward}")
    print(f"指标: {metrics}")
    print("-" * 50)

    # 清理内存
    del agents, reward, metrics
    gc.collect()

    # 3. ε-递减贪婪算法
    def create_decreasing_agent(env: RLEnv, **kwargs) -> BaseAgent:
        return GreedyAgent(
            name="epsilon_decreasing_average",
            env=env,
            greedy_algorithm=epsilon_decreasing_average,
            optimistic_init=ENABLE_OPTIMISTIC,
            optimistic_times=OPTIMISTIC_TIMES,
            **kwargs,
        )

    print("3. 运行ε-递减贪婪算法...")
    agents, reward, metrics = batch_train(
        count=COUNT,
        agent_factory=create_decreasing_agent,
        env=env,
        epsilon_config=epsilon_config,
        steps=STEPS,
        seed=SEED,
    )

    print(f"算法名称: {agents[0].name}")
    print(f"平均奖励: {reward}")
    print(f"指标: {metrics}")
    print("-" * 50)

    # 清理内存
    del agents, reward, metrics
    gc.collect()

    # 4. UCB1算法
    def create_ucb1_agent(env: RLEnv, **kwargs) -> BaseAgent:
        return UCB1Agent(name="ucb1", env=env, **kwargs)

    print("4. 运行UCB1算法...")
    agents, reward, metrics = batch_train(
        count=COUNT,
        agent_factory=create_ucb1_agent,
        env=env,
        epsilon_config=None,  # UCB1不需要ε配置
        steps=STEPS,
        seed=SEED,
    )

    print(f"算法名称: {agents[0].name}")
    print(f"平均奖励: {reward}")
    print(f"指标: {metrics}")
    print("-" * 50)

    print("所有实验完成！")


if __name__ == "__main__":
    run_experiment()
