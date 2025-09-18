"""贪婪算法模块：包含 Pydantic 数据模型

公开接口：
- GreedyRewardsState: 贪婪算法奖励状态模型
"""

from __future__ import annotations

from typing import List

from pydantic import Field

from core.schemas import BaseRewardsState, BaseAlgorithmType
from core.environment import RLEnv


class GreedyRewardsState(BaseRewardsState):
    """贪婪 Agent 获得的奖励记录"""

    q_values: List[float] = Field(default_factory=list, description="每个机器的Q值")
    q_values_optimistic: List[float] = Field(
        default_factory=list, description="每个机器的乐观初始化Q值"
    )

    @classmethod
    def from_env(
        cls: type[GreedyRewardsState],
        env: RLEnv,
        initial_value: float = 0.0,
        initial_count: int = 0,
        optimistic_init: bool = False,
        optimistic_times: int = 1,
    ) -> GreedyRewardsState:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env: 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (int, optional): counts 列表中每个元素的初始值，默认为 0。

        Returns:
            GreedyRewardsState: 一个初始化好的 GreedyRewardsState 实例。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
            q_values=[initial_value] * num_machines,
            q_values_optimistic=[optimistic_times] * num_machines
            if optimistic_init
            else [0] * num_machines,
        )


class GreedyType(BaseAlgorithmType):
    """贪婪算法类型"""

    GREEDY = "greedy"
    EPSILON = "epsilon"
    EPSILON_DECREASING = "epsilon_decreasing"
    GREEDY_ACCUMULATED = "greedy_accumulated"
    EPSILON_ACCUMULATED = "epsilon_accumulated"
    EPSILON_DECREASING_ACCUMULATED = "epsilon_decreasing_accumulated"
