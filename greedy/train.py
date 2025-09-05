from core import GreedyAgent

def train(agent: GreedyAgent, episodes: int = 1000) -> GreedyAgent:
    _printed = False
    for i in range(episodes):
        action = agent.act(epsilon_state=agent.episode_state, epsilon=0.1)
        reward = agent._pull_machine(action)
        agent.rewords.values[action] += reward
        agent.rewords.counts[action] += 1
    
        if agent.episode_state.epsilon <= 0.5 and not _printed:
            print(f"当前 epsilon 已经降到 0.5 了， 回合：{i}")
            _printed = True
    
    total_rewords = sum(agent.rewords.values)
        
    print(f"Name: {agent.name} \nTotal rewards: {total_rewords} \nRewards per machine: {agent.rewords}")
    if agent.name == "epsilon_decreasing_greedy":
        print(f"Final epsilon: {agent.episode_state.epsilon:.4f}")
    print("-" * 50)
    
    return agent