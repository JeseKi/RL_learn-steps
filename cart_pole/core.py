from typing import List
import numpy as np
import math

from schemas import EnvState, CartPoleConfig, Action, Step, \
    Feature, AgentConfig, ActionResult, ActionEvaluation
    
class CartPole:
    def __init__(self, seed: int, cfg: CartPoleConfig) -> None:
        self.np_rnd = np.random.RandomState(seed)
        self.cfg: CartPoleConfig = cfg
        
        # 当前环境状态
        self.state: EnvState = self._env_initialize()
        self.steps: int = 0 # 当前世代步数
        
        # 环境初始化也不需要初始化的变量
        self.total_steps: int = 0 # 总步数
        
    def reset(self) -> EnvState:
        self.steps = 0
        self.state = self._env_initialize()
        return self.state.model_copy()
    
    def step(self, action: Action) -> Step:
        x: float = self.state.x
        x_dot: float = self.state.x_dot
        theta: float = self.state.theta
        theta_dot: float = self.state.theta_dot
        
        force: float = self.cfg.force_mag * action
        
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        temp = (force + self.cfg.pole_mass_length * theta_dot ** 2 * sin_theta) / self.cfg.total_mass
        theta_acc = (self.cfg.gravity * sin_theta - cos_theta * temp) / (
            self.cfg.pole_length * (4.0/3.0 - self.cfg.pole_mass * cos_theta ** 2 / self.cfg.total_mass)
        )
        
        x_acc = temp - self.cfg.pole_mass_length * theta_acc * cos_theta / self.cfg.total_mass
        
        x = x + self.cfg.tau * x_dot
        x_dot = x_dot + self.cfg.tau * x_acc
        theta = theta + self.cfg.tau * theta_dot
        theta_dot = theta_dot + self.cfg.tau * theta_acc
        # 写回最新状态
        self.state = EnvState(x=x, x_dot=x_dot, theta=theta, theta_dot=theta_dot)
        self.steps += 1
        
        terminated = bool(
            x < -self.cfg.x_threshold
            or x > self.cfg.x_threshold
            or theta < -self.cfg.theta_threshold_radians
            or theta > self.cfg.theta_threshold_radians
        )
        truncated = self.steps >= self.cfg.max_steps
        done = terminated or truncated
        if terminated:
            reward = -1.0
        else:
            reward = 1.0 - (abs(theta) / self.cfg.theta_threshold_radians) * 0.5 \
                    - (abs(x) / self.cfg.x_threshold) * 0.5
            reward = max(0.1, reward)
            
        return Step(
            state=self.state.model_copy(),
            reward=reward,
            done=done,
            terminated=terminated,
            truncated=truncated,
            info={},
        )
    
    def _env_initialize(self) -> EnvState:
        state = self.np_rnd.uniform(low=-0.05, high=0.05, size=(4,))
        return EnvState(
            x=state[0],
            x_dot=state[1],
            theta=state[2],
            theta_dot=state[3],
        )
        
class LinearQNet:
    def __init__(self, seed: int, cfg: AgentConfig) -> None:
        self.np_rnd = np.random.RandomState(seed)
        self.cfg = cfg
        
        self.W = self.np_rnd.randn(cfg.n_actions, cfg.phi_dim) * 0.01
        
    def forward(self, phi: Feature) -> np.ndarray:
        """前向传播，返回Q值数组"""
        phi_arr = np.array([
            phi.x, phi.x2, phi.x_dot, phi.x_dot2,
            phi.theta, phi.theta2, phi.theta_dot,
            phi.theta_dot2, phi.bias
        ], dtype=np.float32)
        
        q_values = self.W @ phi_arr  # shape: (n_actions,)
        return q_values

    def predict(self, phi: Feature, epsilon: float = 0) -> ActionResult:
        """预测，返回完整的预测结果"""
        q_values = self.forward(phi)
        
        action_evaluations = [
            ActionEvaluation(action=Action.LEFT, q_value=q_values[0]),
            ActionEvaluation(action=Action.RIGHT, q_value=q_values[1])
        ]
        
        best_action_idx = np.argmax(q_values)
        best_action = Action.LEFT if best_action_idx == 0 else Action.RIGHT
        if self.np_rnd.rand() < epsilon:
            best_action = self.np_rnd.choice([Action.LEFT, Action.RIGHT])
        
        return ActionResult(
            action_space=action_evaluations,
            best_action=best_action
    )

    def exponential_decay(self, total_step: int) -> float:
        """指数衰减退火，一般在训练过程中使用"""
        return self.cfg.eps_end + (self.cfg.eps_start - self.cfg.eps_end) * (self.cfg.eps_rate ** total_step)
    
    def update_td0(self, s: EnvState, a: ActionResult, r: float, s_next: EnvState, done: bool) -> None:
        """使用TD(0)方法更新权重"""
        
        # 1. 将 EnvState 转换为 Feature
        phi_s = self._state_to_feature(s)
        phi_s_next = self._state_to_feature(s_next)
        
        # 2. 计算当前状态-动作的Q值
        q_values_s = self.forward(phi_s)
        
        # 3. 获取执行的动作索引
        action_idx = 0 if a.best_action == Action.LEFT else 1
        q_s_a = q_values_s[action_idx]
        
        # 4. 计算目标Q值
        if done:
            # 终止状态，没有后续奖励
            target_q = r
        else:
            # 非终止状态，需要考虑未来奖励
            q_values_s_next = self.forward(phi_s_next)
            max_q_s_next = np.max(q_values_s_next)  # maxₐ'Q(s',a')
            target_q = r + self.cfg.gamma * max_q_s_next
        
        # 5. 计算TD误差
        td_error = target_q - q_s_a
        
        # 6. 提取特征向量
        phi_arr = np.array([
            phi_s.x, phi_s.x2, phi_s.x_dot, phi_s.x_dot2,
            phi_s.theta, phi_s.theta2, phi_s.theta_dot,
            phi_s.theta_dot2, phi_s.bias
        ], dtype=np.float32)
        
        # 7. 更新权重矩阵对应行
        # ∇Wᵢ = φ(s) * td_error (对于执行的动作i)
        # Wᵢ ← Wᵢ + α * td_error * φ(s)
        self.W[action_idx] += self.cfg.alpha * td_error * phi_arr
    
    def _state_to_feature(self, state: EnvState) -> Feature:
        """将环境状态转换为手工特征"""
        return Feature(
            x=state.x,
            x2=state.x ** 2,
            x_dot=state.x_dot,
            x_dot2=state.x_dot ** 2,
            theta=state.theta,
            theta2=state.theta ** 2,
            theta_dot=state.theta_dot,
            theta_dot2=state.theta_dot ** 2,
            bias=1.0
        )
