from enum import Enum


class DynamicMethod(str, Enum):
    PIECEWISE = "piecewise"
    RANDOM_WALK = "random_walk"
    BOTH = "both"
