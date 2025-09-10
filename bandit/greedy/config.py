"""贪婪算法模块：包含ε-递减相关配置

公开接口：
- EpsilonDecreasingState: ε-递减状态
- EpsilonDecreasingConfig: ε-递减配置
"""

from dataclasses import dataclass


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
