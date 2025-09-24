"""训练模块：提供批训练与单次训练流程

公开接口：
- AverageMetrics: 训练平均指标的 Pydantic 模型
- batch_train: 按批次创建与训练多个 Agent
- train: 训练已构造的 Agent 列表，支持过程数据记录器接入

内部方法：
- _round: 对单个 Agent 进行 steps 次训练，支持基于记录器网格进行采样
- _calculate_averages: 聚合训练结束后的平均指标
"""

from typing import Callable, List, Tuple, Dict, Any
from multiprocessing import Pool
import numpy as np

from pydantic import BaseModel

from core import BaseAgent
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
    num_workers: int = 10,
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
        num_workers (int): 并行训练的进程数量，默认 10
        **kwargs: 传递给代理工厂函数的额外参数

    Returns:
        Tuple[List[BaseAgent], BaseRewardsState, AverageMetrics]: 训练结果
    """
    # 为每个 worker 准备独立的环境种子，避免多进程共享同一随机流
    base_env_seed: int = env.seed
    env_seed_offset = 9973  # 取质数偏移，减少不同索引之间的相关性

    # 准备并行训练的参数列表
    args_list = [
        (
            agent_factory,
            env,
            steps,
            seed + i,
            ((base_env_seed) + env_seed_offset * (i + 1)),
            convergence_threshold,
            convergence_min_steps,
            process_logger,
            kwargs,
        )
        for i in range(count)
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(_run_single_training, args_list)

    # 解包结果：results 是 [(agent1, points1), (agent2, points2), ...]
    agents = []
    all_points = []
    for agent, points in results:
        agents.append(agent)
        all_points.extend(points)

    process_logger._points.extend(all_points)

    avg = _calculate_averages(agents=agents)

    return agents, avg[0], avg[1]


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


def _run_single_training(
    agent_factory: Callable[..., BaseAgent],
    env: RLEnv,
    steps: int,
    seed: int,
    env_seed: int,
    convergence_threshold: float,
    convergence_min_steps: int,
    process_logger: ProcessDataLogger,
    kwargs: Dict[str, Any],
) -> Tuple[BaseAgent, List]:
    """单次训练任务，由 multiprocessing worker 执行

    Args:
        agent_factory: 创建代理的工厂函数
        env: 环境（将在 worker 中被 clone）
        steps: 训练步数
        seed: 代理随机种子
        env_seed: 环境随机种子，确保各 worker 的环境独立
        convergence_threshold: 收敛阈值
        convergence_min_steps: 最小收敛步数
        process_logger: 过程记录器（用于获取配置参数）
        kwargs: 传递给代理工厂函数的额外参数

    Returns:
        Tuple[BaseAgent, List]: 训练完成的代理和收集到的过程数据点
    """
    env_clone = env.clone(seed=env_seed)

    worker_logger = ProcessDataLogger(
        run_id=process_logger.run_id,
        total_steps=steps,
        grid_size=getattr(process_logger, "_grid_size", 100),
    )

    agent_kwargs = {
        "seed": seed,
        "convergence_threshold": convergence_threshold,
        "convergence_min_steps": convergence_min_steps,
        **kwargs,
    }
    agent = agent_factory(env_clone, **agent_kwargs)

    _round(
        agent=agent,
        steps=steps,
        process_logger=worker_logger,
    )

    return agent, worker_logger._points


def _round(
    agent: BaseAgent,
    steps: int,
    process_logger: ProcessDataLogger | None,
):
    _printed: List[bool] = [False, False]
    for i in range(steps):
        action = agent.act(
            epsilon_state=getattr(agent, "episode_state", None),
            epsilon=0.1,
        )
        _ = agent.pull_machine(action)
        # 基于记录器采样或按固定间隔采样（仅在 process_logger 不为 None 时）
        do_record = False
        if process_logger is not None:
            do_record = process_logger.should_record(agent.steps)

        if do_record and process_logger is not None:
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


def _calculate_averages(
    agents: List[BaseAgent],
) -> Tuple[BaseRewardsState, AverageMetrics]:
    """计算平均指标（使用 NumPy 向量化计算）

    Args:
        agents: 训练后的 agents 列表

    Returns:
        Tuple[BaseRewardsState, AverageMetrics]: 平均奖励和平均指标
    """
    if not agents:
        raise ValueError("agents 列表不能为空")

    num_agents = len(agents)
    rewards_list: List[BaseRewardsState] = [agent.rewards for agent in agents]

    # 使用 NumPy 向量化计算平均奖励
    values_array = np.array(
        [r.values for r in rewards_list]
    )  # shape: (num_agents, num_machines)
    counts_array = np.array(
        [r.counts for r in rewards_list]
    )  # shape: (num_agents, num_machines)

    avg_values = values_array.mean(axis=0).tolist()  # 按列求平均
    avg_counts = counts_array.mean(axis=0).tolist()  # 按列求平均

    avg_rewards = BaseRewardsState(values=avg_values, counts=avg_counts)

    # 使用 NumPy 向量化计算平均指标
    metrics_list = [agent.metric() for agent in agents]

    regrets = np.array([m.regret for m in metrics_list])
    regret_rates = np.array([m.regret_rate for m in metrics_list])
    rewards = np.array([sum(m.rewards.values) for m in metrics_list])
    optimal_rates = np.array([m.optimal_rate for m in metrics_list])
    convergence_steps = np.array(
        [getattr(agent, "convergence_steps", 0) for agent in agents]
    )

    avg_regret = regrets.mean()
    avg_regret_rate = regret_rates.mean()
    avg_total_reward = rewards.mean()
    avg_optimal_rate = optimal_rates.mean()
    avg_convergence_steps = convergence_steps.mean()
    avg_convergence_rate = (convergence_steps > 0).sum() / num_agents

    avg_metrics = AverageMetrics(
        avg_regret=avg_regret,
        avg_regret_rate=avg_regret_rate,
        avg_total_reward=avg_total_reward,
        avg_optimal_rate=avg_optimal_rate,
        avg_convergence_steps=avg_convergence_steps,
        avg_convergence_rate=avg_convergence_rate,
    )

    return avg_rewards, avg_metrics
