from typing import List
import numpy as np

from core import CartPole, LinearQNet
from schemas import CartPoleConfig

def evaluate(
    env: CartPole,
    agent: LinearQNet,
    cfg: CartPoleConfig,
    num_episodes: int,
    render: bool = False
) -> dict:
    """
    评估函数（纯利用，无探索）
    
    Args:
        env: CartPole环境
        agent: LinearQNet智能体
        cfg: CartPole配置
        num_episodes: 评估回合数
        render: 是否渲染（如果环境支持的话）
    
    Returns:
        dict: 评估结果统计
    """
    episode_rewards = []
    episode_lengths = []
    
    print(f"开始评估 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        if render:
            print(f"\n评估回合 {episode + 1}")
        
        while steps < cfg.max_steps:
            # 1. 状态转换为特征
            feature = agent._state_to_feature(state)
            
            # 2. 选择最佳动作（无探索）
            action_result = agent.predict(feature, epsilon=0.0)
            
            # 3. 执行动作
            step_result = env.step(action_result.best_action)
            
            # 4. 更新状态和奖励
            state = step_result.state
            total_reward += step_result.reward
            steps += 1
            
            if render:
                print(f"Step {steps}: Action={action_result.best_action.name}, "
                      f"Reward={step_result.reward}, "
                      f"X={state.x:.2f}, Theta={state.theta:.2f}")
            
            # 5. 检查是否结束
            if step_result.done:
                break
        
        # 记录结果
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if (episode + 1) % 10 == 0 or render:
            print(f"评估回合 {episode + 1:3d}: 奖励={total_reward:4.0f}, "
                  f"步数={steps:3d}")
    
    # 计算统计信息
    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    evaluation_results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': float(np.mean(rewards > 0))  # 奖励大于0的成功率
    }
    
    print(f"\n评估完成！")
    print(f"平均奖励: {evaluation_results['mean_reward']:.1f} ± {evaluation_results['std_reward']:.1f}")
    print(f"奖励范围: [{evaluation_results['min_reward']:.0f}, {evaluation_results['max_reward']:.0f}]")
    print(f"平均步数: {evaluation_results['mean_length']:.1f} ± {evaluation_results['std_length']:.1f}")
    
    return evaluation_results