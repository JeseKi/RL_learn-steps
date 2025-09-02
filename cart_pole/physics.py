import math
from typing import Tuple, Optional, Sequence

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


def reward_and_done(
    state: Tuple[float, float, float, float],
    steps_in_episode: int,
    cfg: CartPoleConfig,
    theta_history: Optional[Sequence[float]] = None,
):
    """Reward with stability and persistence shaping.

    - Primary: smaller angle deviation → larger reward.
    - Secondary: boundary proximity is a very small bonus and only when near-upright.
    - Failure penalties: angle failure > boundary failure (single-shot at termination).
    - Persistence: every WINDOW steps, if theta stayed strictly on one side (beyond tol)
      during the whole window → penalty; otherwise a small bonus.
    """
    x, _, theta, _ = state

    # Termination checks
    angle_fail = (theta < -cfg.theta_threshold_radians) or (theta > cfg.theta_threshold_radians)
    pos_fail = (x < -cfg.x_threshold) or (x > cfg.x_threshold)
    terminated = bool(angle_fail or pos_fail)
    truncated = steps_in_episode >= cfg.max_steps
    done = terminated or truncated

    # Primary: angle alignment (1 at perfect, 0 at threshold)
    angle_term = 1.0 - min(1.0, abs(theta) / cfg.theta_threshold_radians)

    # Secondary: boundary proximity — only when near-upright to avoid attracting to edges
    boundary_proximity = 0.0
    if abs(theta) < 0.05:
        boundary_proximity = min(1.0, abs(x) / cfg.x_threshold)

    ANGLE_WEIGHT = 1.0
    BOUNDARY_WEIGHT = 0.05
    ANGLE_PENALTY = 1.0
    BOUNDARY_PENALTY = 0.5

    reward = ANGLE_WEIGHT * angle_term + BOUNDARY_WEIGHT * boundary_proximity
    reward = max(0.0, reward)

    if terminated:
        if angle_fail and not pos_fail:
            reward -= ANGLE_PENALTY
        elif pos_fail and not angle_fail:
            reward -= BOUNDARY_PENALTY
        else:
            reward -= ANGLE_PENALTY
    else:
        # Persistence shaping every WINDOW steps
        WINDOW = 10
        TOL = 0.01
        if theta_history is not None and len(theta_history) >= WINDOW and (steps_in_episode % WINDOW == 0):
            window = theta_history[-WINDOW:]
            all_pos = all(t > TOL for t in window)
            all_neg = all(t < -TOL for t in window)
            if all_pos or all_neg:
                reward -= 0.1
            else:
                reward += 0.05

    return reward, done, terminated, truncated
