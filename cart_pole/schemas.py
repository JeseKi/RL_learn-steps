from typing import List
from pydantic import BaseModel, Field, model_validator
from enum import Enum

class EnvState(BaseModel):
    x: float = Field(default=0, description="小车位置")
    x_dot: float = Field(default=0, description="小车速度")
    theta: float = Field(default=0, description="杆子角度")
    theta_dot: float = Field(default=0, description="杆子角速度")
    
class CartPoleConfig(BaseModel):
    gravity: float = Field(default=9.8, description="重力加速度")
    cart_mass: float = Field(default=1.0, description="小车质量")
    pole_mass: float = Field(default=1, description="杆子质量")
    total_mass: float = Field(default=1.1, description="总质量")
    pole_length: float = Field(default=0.5, description="杆子长度")
    pole_mass_length: float = Field(default=0, description="杆子质量与长度的乘积")
    force_mag: float = Field(default=50.0, description="作用在小车上的力的大小")
    tau: float = Field(default=10e-4, description="采样频率，可以理解为帧率的倒数")
    theta_threshold_radians: float = Field(default=30 * 2 * 3.1415926 / 360, description="杆子最大倾角，超过就算失败")
    x_threshold: float = Field(default=5.0, description="小车最大移动距离，超过就算失败")
    max_steps: int = Field(default=500, description="每一世代的最大步数")
    
    @model_validator(mode='after')
    def compute_pole_mass_length(self):
        self.pole_mass_length = self.pole_mass * self.pole_length
        return self
    
class Step(BaseModel):
    state: EnvState
    reward: float = Field(default=1.0, description="奖励")
    done: bool = Field(default=False, description="是否结束，真正的任务完成/失败")
    terminated: bool = Field(default=False, description="是否终止，比如达到目标，或游戏失败，可能进行重置或下一局等")
    truncated: bool = Field(default=False, description="是否截断，人为截断，比如达到最大步数，时间限制等")
    info: dict = Field(default_factory=dict, description="额外信息")

class Action(int, Enum):
    LEFT = -1
    RIGHT = 1
    
class ActionEvaluation(BaseModel):
    action: Action
    q_value: float = Field(default=0.0, description="动作对应的Q值")
    
class ActionResult(BaseModel):
    action_space: List[ActionEvaluation]
    best_action: Action
    
class Feature(BaseModel):
    x: float = Field(default=0, description="小车位置")
    x2: float = Field(default=0, description="小车位置平方")
    x_dot: float = Field(default=0, description="小车速度")
    x_dot2: float = Field(default=0, description="小车速度平方")
    theta: float = Field(default=0, description="杆子角度")
    theta2: float = Field(default=0, description="杆子角度平方")
    theta_dot: float = Field(default=0, description="杆子角速度")
    theta_dot2: float = Field(default=0, description="杆子角速度平方")
    bias: float = Field(default=1.0, description="偏置项")

class AgentConfig(BaseModel):
    state_dim: int = Field(default=len(EnvState().model_dump()), description="状态维度")
    phi_dim : int = Field(default=len(Feature().model_dump()), description="手工特征维度")
    n_actions: int = Field(default=len(Action), description="动作个数")
    alpha: float = Field(default=0.001, description="学习率")
    gamma: float = Field(default=0.99, description="折扣因子，未来奖励的当前价值")
    eps_start: float = Field(default=1.0, description="epsilon-贪婪策略的初始epsilon，初期高探索，快速了解环境和哪些动作好")
    eps_end: float = Field(default=0.01, description="epsilon-贪婪策略的最终epsilon值，后期低探索，更多利用已学知识")
    eps_rate: float = Field(default=0.9999, description="epsilon-贪婪策略的epsilon衰减速率，控制epsilon从初始值衰减到最终值的速度")
    bias: float = Field(default=1.0, description="手工特征中的偏置项")
