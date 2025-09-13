"""TS 算法模块

公开接口：TSRewardsStaet: TS 算法奖励状态模型
"""

from __future__ import annotations
from typing import cast

from pydantic import Field
import numpy as np

from core.schemas import BaseRewardsState
from core.environment import RLEnv

class TSRewardsState(BaseRewardsState):
    
    alpha: np.ndarray = Field(..., description="每个机器的 alpha 参数")
    beta: np.ndarray = Field(..., description="每个机器的 beta 参数")

    def get_best_machine(self, rng: np.random.Generator) -> int:
        """获取 Beta 采样结果最佳的机器ID

        Returns:
            int: 采样结果最佳的机器ID
        """
        _beta = cast(np.ndarray, rng.beta(self.alpha, self.beta))
        best_machine = int(_beta.argmax())
        return best_machine
            
    @classmethod
    def from_env(
        cls: type[TSRewardsState],
        env: RLEnv,
        initial_value: float = 0.0,
        initial_count: int = 0,
    ) -> TSRewardsState:
        """
        一个类方法，作为自定义的构造函数，用于从 RLEnv 环境初始化。

        Args:
            env: 环境实例。
            initial_value (float, optional): values 列表中每个元素的初始值，默认为 0.0。
            initial_count (int, optional): counts 列表中每个元素的初始值，默认为 0。
        """
        num_machines = len(env.machines)
        return cls(
            values=[initial_value] * num_machines,
            counts=[initial_count] * num_machines,
            alpha=np.array([1] * num_machines, dtype=np.float64),
            beta=np.array([1] * num_machines, dtype=np.float64),
        )