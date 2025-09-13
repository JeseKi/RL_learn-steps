"""训练模块：提供批训练与单次训练流程

公开接口：
- AverageMetrics: 训练平均指标的 Pydantic 模型
- batch_train: 按批次创建与训练多个 Agent
- train: 训练已构造的 Agent 列表，支持过程数据记录器接入

内部方法：
- _round: 对单个 Agent 进行 steps 次训练，支持基于记录器网格进行采样
- _calculate_averages: 聚合训练结束后的平均指标
"""

from typing import Callable, List, Tuple, Optional, Dict, Any

from pydantic import BaseModel

from core.agent import BaseAgent
from core.schemas import BaseRewardsState
from core.environment import RLEnv
from utils.save_data import ProcessDataLogger


class AverageMetrics(BaseModel):
    """平均指标结果"""

    avg_regret: float
    avg_regret_rate: float
    avg_total_reward: float
    avg_optimal_rate: float
    avg_convergence_steps: float
    avg_convergence_rate: float

def batch_train(
    count: int,
    agent_factory: Callable[..., BaseAgent],
    env: RLEnv,
    steps: int,
    seed: int,
    convergence_threshold: float,
    convergence_min_steps: int,
    process_logger: ProcessDataLogger,
    **kwargs,
) -> Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]:
    """批训练 Agent，传入数量，代理工厂函数，环境，步数和初始种子即可训练

    Args:
        count (int): 训练数量
        agent_factory (Callable[..., BaseAgent]): 创建代理的工厂函数
        env (RLEnv): 环境
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
        _agents.append(agent_factory(env, **agent_kwargs))

    agents, reward, metrics = train(
        _agents,
        steps,
        process_logger=process_logger,
    )
    return agents, reward, metrics

def train(
    agents: List[BaseAgent],
    steps: int,
    process_logger: ProcessDataLogger,
) -> Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]:
    """按批量对 agents 进行训练，一般对这些 agents 设置不同的 seed

    Args:
        agents (List[BaseAgent]): 不同 seed 的 agents
        steps (int): 每个 agents 的步数
        log_steps (int): 每个 agents 的日志步数

    Returns:
        Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]: 返回训练后的 agents 和平均后的奖励
    """
    if not agents or not steps:
        raise ValueError("agents 列表或 steps 必须有值")

    _rewards: List[BaseRewardsState] = []
    for agent in agents:
        _round(
            agent=agent,
            steps=steps,
            process_logger=process_logger,
        )
        _rewards.append(agent.rewards)

    avg = _calculate_averages(agents=agents)

    return (agents, avg[0], avg[1])


def _round(
    agent: BaseAgent,
    steps: int,
    process_logger: ProcessDataLogger,
):
    _printed: List[bool] = [False, False]
    for i in range(steps):
        action = agent.act(
            epsilon_state=getattr(agent, "episode_state", None),
            epsilon=0.1,
        )
        _ = agent.pull_machine(action)
        # 基于记录器采样或按固定间隔采样
        do_record = False
        do_record = process_logger.should_record(agent.steps)

        if do_record:
            metrics = agent.metric()
            agent.metrics_history.append(
                (agent.rewards.model_copy(), metrics, agent.steps)
            )
            # 若提供记录器，则追加过程数据点（与指标保存逻辑解耦）
            data: Dict[str, Any] = {
                "agent_name": getattr(agent, "name", "unknown"),
                "seed": getattr(agent, "seed", None),
                "regret": metrics.regret,
                "regret_rate": metrics.regret_rate,
                "total_reward": float(sum(metrics.rewards.values)),
                "optimal_rate": metrics.optimal_rate,
                "convergence_steps": getattr(agent, "convergence_steps", 0),
            }
            process_logger.add(agent.steps, data)

        episode_state = getattr(agent, "episode_state", None)
        if episode_state and episode_state.epsilon <= 0.5 and not _printed[0]:
            _printed[0] = True
        if (
            episode_state
            and episode_state.epsilon <= getattr(episode_state, "min_epsilon", 0.01)
            and not _printed[1]
        ):
            _printed[1] = True


def _calculate_averages(agents: List[BaseAgent]) -> Tuple[BaseRewardsState, AverageMetrics]:
    """计算平均指标

    Args:
        agents: 训练后的 agents 列表

    Returns:
        Tuple[BaseRewardsState, AverageMetrics]: 平均奖励和平均指标
    """
    if not agents:
        raise ValueError("agents 列表不能为空")

    num_agents = len(agents)
    rewards_list = [agent.rewards for agent in agents]

    # 计算平均奖励
    avg_values = [
        sum(values) / num_agents for values in zip(*(r.values for r in rewards_list))
    ]
    avg_counts = [
        sum(counts) / num_agents for counts in zip(*(r.counts for r in rewards_list))
    ]
    avg_rewards = BaseRewardsState(values=avg_values, counts=avg_counts)

    # 计算平均指标
    total_regret = 0.0
    total_regret_rate = 0.0
    total_reward = 0.0
    total_optimal_rate = 0.0
    total_convergence_steps = 0.0
    total_convergence_rate = 0.0

    for agent in agents:
        metrics = agent.metric()
        total_regret += metrics.regret
        total_regret_rate += metrics.regret_rate
        total_reward += sum(metrics.rewards.values)
        total_optimal_rate += metrics.optimal_rate
        convergence_steps = getattr(agent, "convergence_steps", 0)
        total_convergence_steps += convergence_steps
        if convergence_steps > 0:
            total_convergence_rate += 1

    avg_metrics = AverageMetrics(
        avg_regret=total_regret / num_agents,
        avg_regret_rate=total_regret_rate / num_agents,
        avg_total_reward=total_reward / num_agents,
        avg_optimal_rate=total_optimal_rate / num_agents,
        avg_convergence_steps=total_convergence_steps / num_agents,
        avg_convergence_rate=total_convergence_rate / num_agents,
    )

    return avg_rewards, avg_metrics
