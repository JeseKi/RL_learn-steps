"""UCB1算法模块：包含 Pydantic 数据模型

公开接口：
- UCB1RewardsState: UCB1算法奖励状态模型
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass
from pydantic import Field

from core.schemas import BaseRewardsState
from core.environment import RLEnv


@dataclass
class UCBInitState:
    ucb_inited: bool = False
    ucb_inited_index: int = 0


class UCB1RewardsState(BaseRewardsState):
    """UCB1 Agent 获得的奖励记录"""

    q_values: List[float] = Field(default_factory=list, description="每个机器的Q值")
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
        cls: type[UCB1RewardsState],
        env: RLEnv,
        initial_value: float = 0.0,
        initial_count: int = 0,
    ) -> UCB1RewardsState:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env: 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (int, optional): counts 列表中每个元素的初始值，默认为 0。

        Returns:
            UCB1RewardsState: 一个初始化好的 UCB1RewardsState 实例。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
            q_values=[initial_value] * num_machines,
            ucb_values=[0] * num_machines,
        )