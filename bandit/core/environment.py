"""核心模块：包含环境和老虎机类

公开接口：
- SlotMachine: 老虎机类，每次拉动有一定概率获得奖励
- RLEnv: 强化学习环境类，包含多个老虎机
"""

from __future__ import annotations

from typing import List
import random


class SlotMachine:
    """老虎机，每次拉动有一定概率获得奖励"""

    def __init__(self, reward_probability: float, seed: int = 42) -> None:
        self.reward_probability = reward_probability
        self.rng = random.Random(seed)

    def pull(self) -> int:
        return 1 if self.rng.random() < self.reward_probability else 0


class RLEnv:
    """环境中包含多个老虎机，默认为 10 个"""

    def __init__(self, machine_count: int = 10, seed: int = 42) -> None:
        self.machines: List[SlotMachine] = []

        for i in range(machine_count):
            self.machines.append(
                SlotMachine(reward_probability=(i + 1) / (machine_count + 1), seed=seed)
            )

        self.best_reward_machine: SlotMachine = self.machines[
            -1
        ]  # 初始化代码中，对老虎机列表的初始化就是最后一个老虎机是奖励期望最高的

    def pull(self, machine_id: int) -> int:
        assert 0 <= machine_id < len(self.machines)
        return self.machines[machine_id].pull()

    def best_reward(self, steps: int) -> float:
        return self.machines[-1].reward_probability * steps
