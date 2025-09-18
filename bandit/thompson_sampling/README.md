# Thompson Sampling 模块

`thompson_sampling` 模块实现了汤普森采样（Thompson Sampling）算法。这是一种基于贝叶斯推断的算法，通过为每个手臂的奖励概率维护一个后验分布来平衡探索和利用。

## 业务逻辑

汤普森采样的核心思想是“根据后验概率进行决策”。对于每个手臂，算法都会维护一个关于其真实奖励概率的概率分布（后验分布）。在每次需要决策时，算法会从每个手臂的后验分布中随机抽取一个样本，然后选择样本值最大的那个手臂。

在本模块的实现中：

-   **Beta 分布**: 我们假设每个手臂的奖励遵循伯努利分布（即奖励为 0 或 1）。因此，其奖励概率的共轭先验是 Beta 分布。每个手臂的后验分布都由一个 Beta(α, β) 分布来表示。
-   **参数更新**: `α` 和 `β` 的初始值都为 1。当一个手臂被选中后：
    -   如果获得奖励（reward=1），则其 `α` 值加 1。
    -   如果没有获得奖励（reward=0），则其 `β` 值加 1。
-   **决策过程**: 这种机制使得智能体能够根据历史表现动态地调整对每个手臂的“信念”。一个手臂的成功次数越多，其 Beta 分布的样本就越可能偏向于较高的值，从而更有可能被选中。

## 公开接口

### `thompson_sampling.agent.TSAgent`

继承自 `core.BaseAgent`，是汤普森采样算法的智能体实现。

-   `__init__(...)`: 初始化智能体，需要注入 `RLEnv` 和 `TSAlgorithm`。
-   `act() -> int`: 执行一次决策，调用内部算法选择手臂，并与环境交互。
-   `_update_rewards(...)`: 在与环境交互后，根据奖励更新对应手臂的 `alpha` 或 `beta` 值。

### `thompson_sampling.agent.TSAlgorithm`

继承自 `core.BaseAlgorithm`，封装了汤普森采样的决策逻辑。

-   `run() -> int`: 为每个手臂从其对应的 Beta(α, β) 分布中抽取一个样本，然后返回样本值最大的手臂索引。

### `thompson_sampling.schemas.TSRewardsState`

继承自 `core.BaseRewardsState`，用于存储汤普森采样算法的状态。

-   `alpha: np.ndarray`: 存储每个手臂的 Beta 分布的 α 参数。
-   `beta: np.ndarray`: 存储每个手臂的 Beta 分布的 β 参数。

## 数据流

```mermaid
graph TD
    subgraph 训练循环
        A[调用 agent.act()] --> B{TSAlgorithm.run()}
    end

    subgraph TSAlgorithm
        B -- 为每个手臂 --> C[从 Beta(α, β) 分布采样]
        C --> D[选择样本值最大的手臂]
    end

    subgraph TSAgent
        A --> E[调用 env.pull() 获取奖励]
        E -- reward --> F{更新 α 或 β}
    end

    style B fill:#cde4ff
    style C fill:#cde4ff
```

1.  **初始化**: 创建 `TSAgent`，其内部的 `TSRewardsState` 会为每个手臂初始化 `alpha = 1` 和 `beta = 1`。
2.  **决策**: 在训练循环中调用 `agent.act()`，它会触发 `algorithm.run()`。
3.  **采样与选择**: `TSAlgorithm` 对每个手臂，都从其当前的 `Beta(alpha, beta)` 分布中抽取一个随机样本，然后选择样本值最大的手臂。
4.  **环境交互与后验更新**: `TSAgent` 与环境交互获得奖励 `reward`。如果 `reward` 为 1，则更新被选手臂的 `alpha`；如果为 0，则更新 `beta`。这个过程就是贝叶斯推断，即用新的观测数据来更新后验分布。

## 用法示例

```python
from core.environment import RLEnv
from thompson_sampling.agent import TSAgent, TSAlgorithm
from thompson_sampling.schemas import TSAlgorithmType

# 1. 初始化环境
env = RLEnv(machine_count=10, seed=42)

# 2. 初始化算法
algorithm = TSAlgorithm(ts_type=TSAlgorithmType.TS)

# 3. 初始化智能体
agent = TSAgent(
    name="TSAgent",
    env=env,
    algorithm=algorithm,
    seed=42
)

# 4. 运行训练循环
num_steps = 1000
for i in range(num_steps):
    agent.act()

# 5. 查看结果
print(f"训练 {num_steps} 步后:")
# alpha 和 beta 值反映了每个臂的成功/失败计数
print(f"- Alpha 参数: {agent.rewards.alpha}")
print(f"- Beta 参数: {agent.rewards.beta}")
print(f"- 最优臂被选择了 {agent.rewards.counts[env.best_machine_index]} 次")
print(f"- 总后悔值: {agent.regret()}")
```
