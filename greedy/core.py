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
            
        self.best_reward_machine: SlotMachine = self.machines[-1] # 初始化代码中，对老虎机列表的初始化就是最后一个老虎机是奖励期望最高的

    def pull(self, machine_id: int) -> int:
        assert 0 <= machine_id < len(self.machines)
        return self.machines[machine_id].pull()
    
    def best_reward(self, steps: int) -> float:
        return self.machines[-1].reward_probability * steps

@dataclass
class EpsilonDecreasingState:
    epsilon: float
    decay: float
    min_epsilon: float
    
@dataclass
class EpsilonDecreasingConfig:
    start_epsilon: float = 1.0
    decay: float = 0.995
    min_epsilon: float = 0.01
    
@dataclass
class Rewards:
    """贪婪 Agent 获得的奖励记录
    """
    values: List[float] = field(default_factory=list) # 每个机器的累积奖励
    counts: List[float] = field(default_factory=list) # 每个机器的拉动次数
    
    def __init__(self, env: RLEnv):
        """根据传入的环境中的老虎机来初始化列表

        Args:
            env (RLEnv): 环境
        """
        self.values = [0] * len(env.machines)
        self.counts = [0] * len(env.machines)

class GreedyAgent:
    """我们的 Agent 默认使用贪婪算法，来找到最优的老虎机"""
    
    def __init__(
        self, 
        name: str, 
        env: RLEnv, 
        greedy_algorithm: Callable[..., int], 
        epsilon_config: EpsilonDecreasingConfig = EpsilonDecreasingConfig(), 
        seed: int = 42
        ) -> None:
        """贪婪算法的 Agent

        Args:
            name (str): 名称
            env (RLEnv): 环境
            greedy_algorithm (Callable[..., int]): 所使用的贪婪算法
            epsilon_config (EpsilonDecreasingConfig, optional): 退火配置
            seed (int, optional): 种子
        """
        
        self.name = name
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.steps = 0
        self.rewords = Rewards(env)
        self.greedy_algorithm = greedy_algorithm
        self.env = env
        self.episode_state = EpsilonDecreasingState(
            epsilon=epsilon_config.start_epsilon, 
            decay=epsilon_config.decay, 
            min_epsilon=epsilon_config.min_epsilon
            )
        
    def act(self, **kwargs) -> int:
        """选择拉动哪个老虎机，传入一个指定的贪婪算法，根据当前的奖励情况，选择一个老虎机"""
        self.steps += 1
        return self.greedy_algorithm(self.rewords,self.rng, **kwargs)
        
    def regret(self):
        """计算后悔值"""
        regret = self.env.best_reward(self.steps) - sum(self.rewords.values)
        return regret
    
    def regret_rate(self):
        """计算后悔率"""
        return self.regret() / self.env.best_reward(self.steps)
        
    
    def optimal_rate(self):
        return self.rewords.counts[-1] / self.steps if self.steps else 0
        
    def _pull_machine(self, machine_id: int) -> int:
        reward = self.env.pull(machine_id)
        return reward