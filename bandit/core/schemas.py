"""核心模块：包含Pydantic数据模型

公开接口：
- RewardsState: 奖励状态模型
- Metrics: 评估指标模型
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class UCBInitState:
    ucb_inited: bool = False
    ucb_inited_index: int = 0


class RewardsState(BaseModel):
    """贪婪 Agent 获得的奖励记录"""

    values: List[float] = Field(description="每个机器的累积奖励")
    counts: List[float] = Field(description="每个机器的拉动次数")
    optimistic_init: bool = Field(
        default=False,
        description="是否使用乐观初始化：True表示初始Q值为1，False表示初始Q值为0",
    )
    optimistic_times: int = Field(
        default=1,
        description="乐观初始化的次数",
    )
    q_values: List[float] = Field(default_factory=list, description="每个机器的Q值")
    q_values_optimistic: List[float] = Field(
        default_factory=list, description="每个机器的乐观初始化Q值"
    )
    ucb_values: List[float] = Field(default_factory=list, description="每个机器的UCB值")

    _ucb_states: UCBInitState = UCBInitState()

    @property
    def ucb_states(self) -> UCBInitState:
        if not self._ucb_states.ucb_inited:
            for i in range(len(self.counts)):
                if self.counts[i] == 0:
                    self._ucb_states.ucb_inited_index = i
                    return self._ucb_states
            self._ucb_states = UCBInitState(ucb_inited=True)
        return self._ucb_states

    @classmethod
    def from_env(
        cls,
        env,
        initial_value: float = 0.0,
        initial_count: int = 0,
        optimistic_init: bool = False,
        optimistic_times: int = 1,
    ) -> RewardsState:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env: 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (int, optional): counts 列表中每个元素的初始值，默认为 0。

        Returns:
            Rewards: 一个初始化好的 Rewards 实例。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
            optimistic_init=optimistic_init,
            optimistic_times=optimistic_times,
            q_values=[initial_value] * num_machines,
            q_values_optimistic=[optimistic_times] * num_machines,
            ucb_values=[0] * num_machines,
        )


class Metrics(BaseModel):
    regret: float = Field(..., description="后悔值")
    regret_rate: float = Field(..., description="后悔率")
    rewards: RewardsState = Field(..., description="奖励")
    optimal_rate: float = Field(..., description="最佳臂命中率")
