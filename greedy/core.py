from dataclasses import dataclass, field
from typing import Callable, List
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
    def __init__(self, machine_count: int = 10, seed:int = 42) -> None:
        self.machines: List[SlotMachine] = []
        
        for i in range(machine_count):
            self.machines.append(
                SlotMachine(
                        reward_probability=(i + 1) / (machine_count + 1), 
                        seed=seed
                        )
                    )

    def pull(self, machine_id: int) -> int:
        assert 0 <= machine_id < len(self.machines)
        return self.machines[machine_id].pull()

@dataclass
class EpsilonDecreasingState:
    epsilon: float
    decay: float
    min_epsilon: float
    
@dataclass
class Rewards:
    values: List[int] = field(default_factory=list)
    counts: List[int] = field(default_factory=list)
    
    def __init__(self, env: RLEnv):
        self.values = [0] * len(env.machines)
        self.counts = [0] * len(env.machines)

class GreedyAgent:
    """我们的 Agent 默认使用贪婪算法，来找到最优的老虎机"""
    
    def __init__(self, name: str, env: RLEnv, greedy_algorithm: Callable[..., int], seed: int = 42) -> None:
        self.name = name
        self.rng = random.Random(seed)
        self.rewords = Rewards(env)
        self.greedy_algorithm = greedy_algorithm
        self.env = env
        self.episode_state = EpsilonDecreasingState(epsilon=1, decay=0.995, min_epsilon=0.01)
        
    def act(self, **kwargs) -> int:
        """选择拉动哪个老虎机，传入一个指定的贪婪算法，根据当前的奖励情况，选择一个老虎机"""
        return self.greedy_algorithm(self.rewords,self.rng, **kwargs)
        
    def _pull_machine(self, machine_id: int) -> int:
        reward = self.env.pull(machine_id)
        return reward