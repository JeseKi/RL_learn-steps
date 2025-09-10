from dataclasses import dataclass
from typing import List, Tuple

from core import GreedyAgent, RewardsState


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
    agents: List[GreedyAgent], steps: int
) -> Tuple[List[GreedyAgent], RewardsState, AverageMetrics]:
    """按批量对 agents 进行训练，一般对这些 agents 设置不同的 seed

    Args:
        agents (List[GreedyAgent]): 不同 seed 的 agents
        steps (int): 每个 agents 的步数

    Returns:
        Tuple[List[GreedyAgent], Rewards]: 返回训练后的 agents 和平均后的奖励
    """
    if not agents or not steps:
        raise ValueError("agents 列表或 steps 必须有值")

    _rewards: List[RewardsState] = []
    for agent in agents:
        _round(agent=agent, steps=steps)
        _rewards.append(agent.rewards)

    avg = _calculate_averages(agents=agents)

    return (agents, avg[0], avg[1])


def _round(agent: GreedyAgent, steps: int):
    _printed: List[bool] = [False, False]
    for _ in range(steps):
        action = agent.act(epsilon_state=agent.episode_state, epsilon=0.1, steps=agent.steps - 1)
        _ = agent.pull_machine(action)
        agent.metrics_history.append(
            (agent.rewards.model_copy(), agent.metric(), agent.steps - 1)
        )

        if agent.episode_state.epsilon <= 0.5 and not _printed[0]:
            # print(f"当前 epsilon 已经降到 0.5 了， 回合：{i}")
            _printed[0] = True
        if (
            agent.episode_state.epsilon <= agent.episode_state.min_epsilon
            and not _printed[1]
        ):
            # print(f"当前 epsilon 已经降到 {agent.episode_state.min_epsilon} 了， 回合：{i}")
            _printed[1] = True

    # total_rewords = sum(agent.rewords.values)

    # print(f"Name: {agent.name} \nTotal rewards: {total_rewords} \nRewards per machine: {agent.rewords}")
    # if agent.name == "epsilon_decreasing_greedy":
    #     print(f"Final epsilon: {agent.episode_state.epsilon:.4f}")
    # print("-" * 50)


def _calculate_averages(agents: List[GreedyAgent]) -> Tuple[RewardsState, AverageMetrics]:
    """计算平均指标

    Args:
        agents: 训练后的 agents 列表

    Returns:
        Tuple[Rewards, AverageMetrics]: 平均奖励和平均指标
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
    avg_rewards = RewardsState(values=avg_values, counts=avg_counts)

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
        total_convergence_steps += agent.convergence_steps
        if agent.convergence_steps > 0:
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
