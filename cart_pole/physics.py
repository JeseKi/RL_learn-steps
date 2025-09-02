import math
from typing import Tuple

from schemas import CartPoleConfig

def step_dynamics(state: Tuple[float, float, float, float], force: float, cfg: CartPoleConfig) -> Tuple[float, float, float, float]:
    """Single physics step using the same equations as training.

    state: (x, x_dot, theta, theta_dot)
    force: external force (N), typically +/- cfg.force_mag
    cfg:   CartPoleConfig
    returns next (x, x_dot, theta, theta_dot)
    """
    x, x_dot, theta, theta_dot = state

    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    temp = (force + cfg.pole_mass_length * theta_dot * theta_dot * sin_t) / cfg.total_mass
    theta_acc = (cfg.gravity * sin_t - cos_t * temp) / (
        cfg.pole_length * (4.0 / 3.0 - (cfg.pole_mass * cos_t * cos_t) / cfg.total_mass)
    )
    x_acc = temp - (cfg.pole_mass_length * theta_acc * cos_t) / cfg.total_mass

    # Euler integration with cfg.tau
    x = x + cfg.tau * x_dot
    x_dot = x_dot + cfg.tau * x_acc
    theta = theta + cfg.tau * theta_dot
    theta_dot = theta_dot + cfg.tau * theta_acc

    return (x, x_dot, theta, theta_dot)


def reward_and_done(state: Tuple[float, float, float, float], steps_in_episode: int, cfg: CartPoleConfig):
    """Compute reward and termination/truncation in the same way as training env."""
    x, _, theta, _ = state
    terminated = bool(
        x < -cfg.x_threshold
        or x > cfg.x_threshold
        or theta < -cfg.theta_threshold_radians
        or theta > cfg.theta_threshold_radians
    )
    truncated = steps_in_episode >= cfg.max_steps
    done = terminated or truncated

    if terminated:
        reward = -1.0
    else:
        reward = 1.0 - (abs(theta) / cfg.theta_threshold_radians) * 0.5 \
                - (abs(x) / cfg.x_threshold) * 0.5
        reward = max(0.1, reward)

    return reward, done, terminated, truncated

