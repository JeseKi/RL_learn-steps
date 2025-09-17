"""核心模块：包含环境和老虎机类

公开接口：
- SlotMachine: 老虎机类，每次拉动有一定概率获得奖励
- RLEnv: 强化学习环境类，包含多个老虎机
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List
import random

import numpy as np
from .schemas import PiecewizeMethod

if TYPE_CHECKING:
    from .schemas import PiecewizeMethod


class SlotMachine:
    """老虎机，每次拉动有一定概率获得奖励"""

    def __init__(self, reward_probability: float, seed: int = 42) -> None:
        self.reward_probability = reward_probability
        self.rng = np.random.default_rng(seed)

    def pull(self) -> int:
        return 1 if self.rng.random() < self.reward_probability else 0


class RLEnv:
    """环境中包含多个老虎机，默认为 10 个"""

    def __init__(
        self, 
        machine_count: int = 10,
        random_walk_internal: int = 0,
        random_walk_machine_num: int = 0,
        piecewise_internal: int = 0,
        piecewize_method: PiecewizeMethod = PiecewizeMethod.UPSIDE_DOWN,
        seed: int = 42
        ) -> None:
        self.machines: List[SlotMachine] = []
        self.machine_count = machine_count
        self.seed = seed
        
        self.rng = random.Random(seed)
        self.nprng = np.random.default_rng(seed)

        self._reset(machine_count, seed)
        
        self.best_reward_machine: SlotMachine = max(
            self.machines, key=lambda machine: machine.reward_probability
        )
        self.best_machine_index: int = self.machines.index(self.best_reward_machine)
        
        self.piecewize_method: PiecewizeMethod = piecewize_method
        self.random_walk_internal = random_walk_internal
        self.random_walk_machine_num = random_walk_machine_num
        self.piecewise_internal = piecewise_internal

    def pull(self, machine_id: int, steps: int) -> int:
        try: 
            assert 0 <= machine_id < len(self.machines)
        except AssertionError:
            raise ValueError(f"机器ID超出范围: {machine_id}")
        if self.random_walk_internal > 0 and steps % self.random_walk_internal == 0:
            self._random_walk_reward()
        if self.piecewise_internal > 0 and steps % self.piecewise_internal == 0:
            self._piecewize_reward()
        
        return self.machines[machine_id].pull()

    def best_reward(self, steps: int) -> float:
        return self.best_reward_machine.reward_probability * steps
    
    def _random_walk_reward(self) -> None:
        m = self.rng.sample(self.machines, self.random_walk_machine_num)
        samples = self.nprng.normal(0, 1, self.random_walk_machine_num)
        for machine, sample in zip(m, samples):
            r = machine.reward_probability + sample
            if r < 0 or r > 1:
                r = machine.reward_probability - sample
            machine.reward_probability = r

    def _piecewize_reward(self) -> None:
        if self.piecewize_method == PiecewizeMethod.UPSIDE_DOWN:
            self._upside_down()
        elif self.piecewize_method == PiecewizeMethod.RESET:
            self._reset(self.machine_count, self.seed)
        elif self.piecewize_method == PiecewizeMethod.ADDITION_SUBTRACTION:
            self._addition_subtraction()
    
    def _upside_down(self) -> None:
        machines_sorted = sorted(self.machines, key=lambda m: m.reward_probability)
        n = len(machines_sorted)
        
        for i, machine in enumerate(machines_sorted):
            machine.reward_probability = (n - 1 - i) / n
    
    def _reset(self, machine_count: int, seed: int) -> None:
        self.machines = []
            
        for i in range(machine_count):
            self.machines.append(
                SlotMachine(reward_probability=(i + 1) / (machine_count + 1), seed=seed)
            )

        self.rng.shuffle(self.machines)
    
    def _addition_subtraction(self) -> None:
        for machine in self.machines:
            r = machine.reward_probability + 0.5
            if r > 1:
                r = machine.reward_probability - 0.5
            machine.reward_probability = r