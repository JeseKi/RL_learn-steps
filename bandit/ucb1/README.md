# UCB1 模块

`ucb1` 模块实现了 UCB1 (Upper Confidence Bound 1) 算法。该算法基于“乐观面对不确定性”的原则，通过计算每个手臂的置信区间上界来有效地平衡探索（Exploration）和利用（Exploitation）。

## 业务逻辑

UCB1 算法的核心思想是为每个手臂计算一个分数，该分数是其当前平均奖励（利用价值）和其不确定性（探索价值）的和。在每次决策时，算法会选择分数最高的手臂。

该分数由 UCB1 公式定义：

```
UCB1(i) = Q(i) + sqrt(2 * ln(t) / N(i))
```

其中：
-   `Q(i)`: 手臂 `i` 的当前平均奖励（Q-value）。
-   `t`: 当前的总步数。
-   `N(i)`: 手臂 `i` 到目前为止被选择的次数。

这个公式的第二部分 `sqrt(2 * ln(t) / N(i))` 是“探索项”。随着总步数 `t` 的增加，或者当某个手臂被选择的次数 `N(i)` 较少时，这个项的值会变大，从而增加了该手臂被选中的概率。这保证了即使是当前表现不佳的手臂也能获得被探索的机会。

在实现上，算法会首先确保每个手臂都被至少选择一次，以获得初始的 Q-value，之后再应用 UCB1 公式进行决策。

## 公开接口

### `ucb1.agent.UCBAgent`

继承自 `core.BaseAgent`，是 UCB1 算法的智能体实现。

-   `__init__(...)`: 初始化智能体，需要注入 `RLEnv` 和 `UCB1Algorithm`。
-   `act() -> int`: 执行一次决策，调用内部的 UCB1 算法选择手臂，并与环境交互。
-   `_update_q_value(...)`: 在获得奖励后，通过增量方式更新手臂的 Q-value。

### `ucb1.agent.UCB1Algorithm`

继承自 `core.BaseAlgorithm`，封装了 UCB1 的决策逻辑。

-   `run() -> int`: 首先处理初始化阶段（确保每个手臂都被拉动一次），然后根据 UCB1 公式计算所有手臂的分数，并返回分数最高的手臂索引。

### `ucb1.schemas.UCB1RewardsState`

继承自 `core.BaseRewardsState`，用于存储 UCB1 算法的状态。

-   `q_values: np.ndarray`: 存储每个手臂的 Q-value。
-   `ucb_values: np.ndarray`: 存储每一步计算出的各手臂的 UCB 分数。

## 数据流

```mermaid
graph TD
    subgraph 初始化阶段
        A[循环拉动每个手臂一次] --> B[建立初始 Q-value]
    end

    subgraph 决策阶段 (t > K)
        C[调用 agent.act()] --> D{UCB1Algorithm.run()}
    end
    
    subgraph UCB1Algorithm
        D -- 对每个手臂 --> E[计算 UCB 分数]
        E --> F[选择分数最高的手臂]
    end

    subgraph UCBAgent
        C --> G[调用 env.pull() 获取奖励]
        G --> H[更新 Q-value 和计数]
    end

    A --> C
    style D fill:#cde4ff
    style E fill:#cde4ff
```

1.  **初始化**: `UCB1Algorithm` 首先会确保每个手臂都被选择一次，以避免在 UCB1 公式的分母中出现零。
2.  **决策**: 初始化完成后，每次调用 `agent.act()` 时，`algorithm.run()` 会被触发。
3.  **计算 UCB 分数**: `UCB1Algorithm` 使用 UCB1 公式为每个手臂计算一个分数。
4.  **选择与交互**: 选择分数最高的手臂，然后 `UCBAgent` 与环境交互获得奖励。
5.  **状态更新**: `UCBAgent` 根据奖励更新被选手臂的 `q_value` 和 `count`，为下一次决策做准备。

## 用法示例

```python
from core.environment import RLEnv
from ucb1.agent import UCBAgent, UCB1Algorithm
from ucb1.schemas import UCB1AlgorithmType

# 1. 初始化环境
env = RLEnv(machine_count=10, seed=42)

# 2. 初始化算法
algorithm = UCB1Algorithm(ucb1_type=UCB1AlgorithmType.UCB1)

# 3. 初始化智能体
agent = UCBAgent(
    name="UCB1Agent",
    env=env,
    algorithm=algorithm,
    seed=42
)

# 4. 运行训练循环
num_steps = 1000
for _ in range(num_steps):
    agent.act()

# 5. 查看结果
print(f"训练 {num_steps} 步后:")
print(f"- 最终的 Q-values: {agent.rewards.q_values}")
print(f"- 最优臂被选择了 {agent.rewards.counts[env.best_machine_index]} 次")
print(f"- 总后悔值: {agent.regret()}")
```
