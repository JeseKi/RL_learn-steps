from dataclasses import dataclass
from typing import List, Tuple

from core.agent import BaseAgent
from core.schemas import BaseRewardsState


@dataclass
class AverageMetrics:
    """平均指标结果"""

    avg_regret: float
    avg_regret_rate: float
    avg_total_reward: float
    avg_optimal_rate: float
    avg_convergence_steps: float
    avg_convergence_rate: float


def train(
    agents: List[BaseAgent], steps: int
) -> Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]:
    """按批量对 agents 进行训练，一般对这些 agents 设置不同的 seed

    Args:
        agents (List[BaseAgent]): 不同 seed 的 agents
        steps (int): 每个 agents 的步数

    Returns:
        Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]: 返回训练后的 agents 和平均后的奖励
    """
    if not agents or not steps:
        raise ValueError("agents 列表或 steps 必须有值")

    _rewards: List[BaseRewardsState] = []
    for agent in agents:
        _round(agent=agent, steps=steps)
        _rewards.append(agent.rewards)

    avg = _calculate_averages(agents=agents)

    return (agents, avg[0], avg[1])


def _round(agent: BaseAgent, steps: int):
    _printed: List[bool] = [False, False]
    for _ in range(steps):
        action = agent.act(
            epsilon_state=getattr(agent, "episode_state", None), epsilon=0.1
        )
        _ = agent.pull_machine(action)
        agent.metrics_history.append(
            (agent.rewards.model_copy(), agent.metric(), agent.steps - 1)
        )

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
