from typing import List
import numpy as np

from core import CartPole, LinearQNet
from schemas import CartPoleConfig

def train(
    env: CartPole,
    agent: LinearQNet,
    cfg: CartPoleConfig,
    num_episodes: int,
    log_interval: int = 100,
) -> List[float]:
    """
    训练函数
    
    Args:
        env: CartPole环境
        agent: LinearQNet智能体
        cfg: CartPole配置
        num_episodes: 训练回合数
        max_steps_per_episode: 每回合最大步数，如果为None则使用cfg中的设置
    
    Returns:
        List[float]: 每回合的奖励列表
    """
    episode_rewards = []  # 记录每回合奖励
    total_steps = 0       # 总步数，用于epsilon退火
    
    print(f"开始训练 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0.0
        steps_in_episode = 0
        
        while steps_in_episode < cfg.max_steps:
            # 1. 状态转换为特征
            feature = agent._state_to_feature(state)
            
            # 2. 获取epsilon（指数退火）
            epsilon = agent.exponential_decay(total_steps)
            
            # 3. 选择动作
            action_result = agent.predict(feature, epsilon)
            
            # 4. 执行动作
            step_result = env.step(action_result.best_action)
            
            # 5. TD更新
            agent.update_td0(
                s=state,
                a=action_result,
                r=step_result.reward,
                s_next=step_result.state,
                done=step_result.done
            )
            
            # 6. 更新状态和奖励
            state = step_result.state
            total_reward += step_result.reward
            total_steps += 1
            steps_in_episode += 1
            
            # 7. 检查是否结束
            if step_result.done:
                break
        
        # 记录回合结果
        episode_rewards.append(total_reward)
        
        # 打印进度
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            current_epsilon = agent.exponential_decay(total_steps)
            print(f"Episode {episode + 1:4d}: "
                  f"Average Reward (last {log_interval}): {avg_reward:6.1f}, "
                  f"Epsilon: {current_epsilon:.3f}")
    
    print(f"训练完成！总步数: {total_steps}")
    return episode_rewards