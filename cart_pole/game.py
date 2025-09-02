# cartpole_game.py
# 倒立摆小游戏 (Pygame)
# 操作：← / →：对小车施加力；P：暂停/继续；R：重置；Esc：退出
# 目标：让摆杆尽量保持竖直，别让小车出界或摆角过大

import math
from pathlib import Path
import random
import pygame

# ========================
# 物理参数（单位制：m, s, N）
# ========================
G = 9.8                 # 重力加速度
M_CART = 1.0            # 小车质量
M_POLE = 0.1            # 摆杆质量
TOTAL_MASS = M_CART + M_POLE
L = 0.5                 # 摆杆“半长”（从轴到杆端的距离，米）
POLEMASS_LENGTH = M_POLE * L

# 输入力大小（越大操控越“硬”，可自行微调）
FORCE_MAG = 12.0

# 出界与失败判定阈值
X_THRESHOLD = 2.4       # 小车位置阈值（米）
THETA_THRESHOLD = math.radians(24)  # 摆角阈值（弧度），从竖直方向起算

# 数值积分步长（秒），与 FPS 同步更稳定
FPS = 120
DT = 1.0 / FPS

# ========================
# 画面与缩放
# ========================
SCREEN_W, SCREEN_H = 900, 600
GROUND_Y = int(SCREEN_H * 0.75)

WORLD_W = X_THRESHOLD * 2.0 * 1.2  # 视野范围稍大于阈值
PX_PER_M = (SCREEN_W * 0.7) / (X_THRESHOLD * 2.0)  # 像素/米

# 小车外观尺寸（米）
CART_W = 0.5
CART_H = 0.25

# 颜色
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (120, 120, 120)
BLUE = (70, 120, 255)
GREEN = (60, 180, 120)
RED = (230, 60, 60)

# ========================
# 工具函数
# ========================
def m2px(x_m):
    """将世界坐标（米）X 映射到屏幕像素坐标"""
    screen_center = SCREEN_W // 2
    return int(screen_center + x_m * PX_PER_M)

def draw_cart_pole(surf, x, theta):
    """绘制小车与摆杆。theta 为从竖直方向（向上）起算的弧度，右手正向为顺时针"""
    # 小车像素尺寸
    cart_w_px = int(CART_W * PX_PER_M)
    cart_h_px = int(CART_H * PX_PER_M)
    cart_x_px = m2px(x)
    cart_y_px = GROUND_Y - cart_h_px

    # 画地面
    pygame.draw.line(surf, GRAY, (0, GROUND_Y), (SCREEN_W, GROUND_Y), 2)

    # 画小车底座
    rect = pygame.Rect(cart_x_px - cart_w_px // 2, cart_y_px, cart_w_px, cart_h_px)
    pygame.draw.rect(surf, BLUE, rect, border_radius=10)

    # 车轮
    wheel_r = max(6, cart_h_px // 6)
    pygame.draw.circle(surf, BLACK, (rect.left + cart_w_px // 4, GROUND_Y + wheel_r), wheel_r)
    pygame.draw.circle(surf, BLACK, (rect.right - cart_w_px // 4, GROUND_Y + wheel_r), wheel_r)

    # 摆杆
    pivot = (cart_x_px, cart_y_px)  # 以小车顶面中心为铰接点
    pole_len_px = int((L * 2.0) * PX_PER_M)  # L 是半长，绘制用全长
    # theta=0 表示竖直向上；端点坐标：
    end_x = pivot[0] + int(pole_len_px * math.sin(theta))
    end_y = pivot[1] - int(pole_len_px * math.cos(theta))

    # 杆与铰点
    pygame.draw.line(surf, GREEN, pivot, (end_x, end_y), 6)
    pygame.draw.circle(surf, BLACK, pivot, 6)

def step_dynamics(state, force, dt):
    """
    经典 CartPole 连续动力学（无摩擦）：
    state: (x, x_dot, theta, theta_dot)
    force: 外力 N
    返回下一个状态
    """
    x, x_dot, theta, theta_dot = state

    # 来自 OpenAI Gym(CartPole) 的解析公式
    # 参考：temp 与两个加速度
    # 注意：theta 是相对竖直方向的角度
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sin_t) / TOTAL_MASS
    theta_acc = (G * sin_t - cos_t * temp) / (L * (4.0/3.0 - (M_POLE * cos_t * cos_t) / TOTAL_MASS))
    x_acc = temp - (POLEMASS_LENGTH * theta_acc * cos_t) / TOTAL_MASS

    # 简单的轻微阻尼，提升可玩性（可注释掉）
    cart_damping = 0.05
    pole_damping = 0.002
    x_acc -= cart_damping * x_dot
    theta_acc -= pole_damping * theta_dot

    # 欧拉积分
    x += x_dot * dt
    x_dot += x_acc * dt
    theta += theta_dot * dt
    theta_dot += theta_acc * dt

    return (x, x_dot, theta, theta_dot)

def is_failed(state):
    x, _, theta, _ = state
    return (abs(x) > X_THRESHOLD) or (abs(theta) > THETA_THRESHOLD)

def reset_state():
    # 以接近竖直的小扰动初始化
    x = 0.0
    x_dot = 0.0
    theta = random.uniform(-0.05, 0.05)  # ±~3°
    theta_dot = 0.0
    return (x, x_dot, theta, theta_dot)

# ========================
# 游戏主循环
# ========================
def main():
    pygame.init()
    pygame.display.set_caption("倒立摆：Cart-Pole (Pygame)")
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    font_small = pygame.font.Font(Path.cwd() / "assets"/ "微软雅黑.ttf", 18)
    font_med   = pygame.font.Font(Path.cwd() / "assets"/ "微软雅黑.ttf", 24)
    font_big   = pygame.font.Font(Path.cwd() / "assets"/ "微软雅黑.ttf", 36)

    state = reset_state()
    running = True
    paused = False
    game_over = False
    time_alive = 0.0
    best_time = 0.0

    # 输入状态（持续施力）
    force_left = False
    force_right = False

    while running:
        dt_ms = clock.tick(FPS)
        dt = DT  # 将物理步长固定为 DT，画面用 FPS 控制
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    # 立即重置
                    state = reset_state()
                    game_over = False
                    time_alive = 0.0
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    force_left = True
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    force_right = True

            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    force_left = False
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    force_right = False

        screen.fill(WHITE)

        # 物理更新
        if not paused and not game_over:
            force = 0.0
            if force_left and not force_right:
                force = -FORCE_MAG
            elif force_right and not force_left:
                force = FORCE_MAG

            state = step_dynamics(state, force, dt)
            time_alive += dt

            if is_failed(state):
                game_over = True
                best_time = max(best_time, time_alive)

        # 绘制倒立摆
        x, x_dot, theta, theta_dot = state
        draw_cart_pole(screen, x, theta)

        # HUD
        info_lines = [
            f"Time: {time_alive:6.2f}s   Best: {best_time:6.2f}s",
            f"x={x:+.2f} m   x_dot={x_dot:+.2f} m/s",
            f"theta={math.degrees(theta):+6.2f} deg   theta_dot={math.degrees(theta_dot):+6.2f} deg/s",
            "Controls: ←/→ or A/D=施加推力,  P=暂停,  R=重置,  Esc=退出",
        ]
        for i, line in enumerate(info_lines):
            text = font_small.render(line, True, BLACK)
            screen.blit(text, (16, 16 + i * 22))

        # 边界提示
        # 画出 X 阈值位置的标记线，帮助玩家预判出界
        left_x = m2px(-X_THRESHOLD)
        right_x = m2px(X_THRESHOLD)
        pygame.draw.line(screen, RED, (left_x, GROUND_Y + 2), (left_x, GROUND_Y - 80), 2)
        pygame.draw.line(screen, RED, (right_x, GROUND_Y + 2), (right_x, GROUND_Y - 80), 2)

        if paused:
            msg = font_big.render("PAUSED", True, RED)
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, 80))

        if game_over:
            over = font_big.render("GAME OVER", True, RED)
            tip = font_med.render("按 R 重置；继续练手以刷新最佳时间！", True, BLACK)
            screen.blit(over, (SCREEN_W // 2 - over.get_width() // 2, 120))
            screen.blit(tip, (SCREEN_W // 2 - tip.get_width() // 2, 170))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
