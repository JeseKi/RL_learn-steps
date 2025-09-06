from dataclasses import dataclass
from typing import List, Tuple

from core import GreedyAgent, Rewards


@dataclass
class AverageMetrics:
    """平均指标结果"""

    avg_regret: float
    avg_regret_rate: float
    avg_total_reward: float
    avg_optimal_rate: float


def train(
    agents: List[GreedyAgent], steps: int
) -> Tuple[List[GreedyAgent], Rewards, AverageMetrics]:
    """按批量对 agents 进行训练，一般对这些 agents 设置不同的 seed

    Args:
        agents (List[GreedyAgent]): 不同 seed 的 agents
        steps (int): 每个 agents 的步数

    Returns:
        Tuple[List[GreedyAgent], Rewards]: 返回训练后的 agents 和平均后的奖励
    """
    if not agents or not steps:
        raise ValueError("agents 列表或 steps 必须有值")

    _rewards: List[Rewards] = []
    for agent in agents:
        _round(agent=agent, steps=steps)
        _rewards.append(agent.rewords)

    return (
        agents,
        _average_rewards(rewards_list=_rewards),
        _average_metrics_structured(agents=agents),
    )


def _round(agent: GreedyAgent, steps: int):
    _printed: List[bool] = [False, False]
    for i in range(steps):
        action = agent.act(epsilon_state=agent.episode_state, epsilon=0.1)
        reward = agent._pull_machine(action)
        agent.rewords.values[action] += reward
        agent.rewords.counts[action] += 1

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


def _average_rewards(rewards_list: List[Rewards]) -> Rewards:
    if not rewards_list:
        raise ValueError("不能传入一个空列表")

    num_agents = len(rewards_list)
    num_machines = len(rewards_list[0].values)

    avg_values: List[float] = [
        sum(r.values[i] for r in rewards_list) / num_agents for i in range(num_machines)
    ]
    avg_counts: List[float] = [
        sum(r.counts[i] for r in rewards_list) / num_agents for i in range(num_machines)
    ]

    avg_reward = Rewards.__new__(Rewards)
    avg_reward.values = avg_values
    avg_reward.counts = avg_counts
    return avg_reward


def _average_metrics_structured(agents: List[GreedyAgent]) -> AverageMetrics:
    """计算 agents 的平均指标并返回结构化结果

    Args:
        agents (List[GreedyAgent]): 训练后的 agents 列表

    Returns:
        AverageMetrics: 平均指标结果
    """
    if not agents:
        return AverageMetrics(0.0, 0.0, 0.0, 0.0)

    num_agents = len(agents)

    avg_regret = sum(agent.regret() for agent in agents) / num_agents
    avg_regret_rate = sum(agent.regret_rate() for agent in agents) / num_agents
    avg_total_reward = sum(sum(agent.rewords.values) for agent in agents) / num_agents
    avg_optimal_rate = sum(agent.optimal_rate() for agent in agents) / num_agents

    return AverageMetrics(
        avg_regret=avg_regret,
        avg_regret_rate=avg_regret_rate,
        avg_total_reward=avg_total_reward,
        avg_optimal_rate=avg_optimal_rate,
    )
