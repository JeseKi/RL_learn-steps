# Greedy 模块

`greedy` 模块提供了 Epsilon-Greedy (ε-贪婪) 算法的具体实现。它基于 `core` 模块的抽象基类，定义了 `GreedyAgent` 和 `GreedyAlgorithm` 来解决多臂老虎机问题。

## 业务逻辑

本模块的核心是实现经典的“探索与利用”（Exploration vs. Exploitation）权衡策略。`GreedyAgent` 通过增量更新的方式计算每个手臂的 Q-value（预期奖励），而 `GreedyAlgorithm` 则根据不同的策略来选择手臂。

支持的算法策略包括：

1.  **纯贪婪 (Greedy)**: 总是选择当前估计回报最高的手臂（纯利用）。
2.  **ε-贪婪 (Epsilon-Greedy)**: 以 ε 的概率随机选择一个手臂（探索），以 1-ε 的概率选择当前最优的手臂（利用）。
3.  **ε-递减贪婪 (Epsilon-Decreasing Greedy)**: ε 的值会随着训练步数增加而衰减，实现从偏向探索到偏向利用的平滑过渡。
4.  **乐观初始化 (Optimistic Initialization)**: 将所有手臂的初始 Q-value 设置为一个较高的值，以鼓励智能体在早期尝试所有手臂，从而进行有效的探索。

这些策略可以通过 `GreedyType` 枚举进行选择。

## 公开接口

### `greedy.agent.GreedyAgent`

继承自 `core.BaseAgent`，是 ε-贪婪算法的智能体实现。

-   `__init__(...)`: 初始化智能体，需要注入 `RLEnv` 和 `GreedyAlgorithm`。
-   `act() -> int`: 执行一次决策，调用内部算法选择手臂，并与环境交互。
-   `_update_q_value(...)`: 内部方法，使用增量公式 `Q(A) ← Q(A) + (R - Q(A)) / N(A)` 来更新 Q-value。

### `greedy.agent.GreedyAlgorithm`

继承自 `core.BaseAlgorithm`，封装了所有贪婪策略的决策逻辑。

-   `__init__(greedy_type: GreedyType, optimistic_init: bool, ...)`: 初始化算法，需要指定 `GreedyType` 以及是否启用乐观初始化。
-   `run() -> int`: 根据设定的 `GreedyType`（如 `EPSILON`）执行决策逻辑，返回被选中的手臂索引。

### `greedy.schemas.GreedyType`

一个枚举类，用于定义 `GreedyAlgorithm` 所使用的具体策略。可选值包括：
-   `GREEDY`
-   `EPSILON`
-   `EPSILON_DECREASING`
-   `GREEDY_ACCUMULATED` (基于累计奖励而非 Q-value)
-   `EPSILON_ACCUMULATED`
-   `EPSILON_DECREASING_ACCUMULATED`

### `greedy.config.EpsilonDecreasingConfig`

一个数据类，用于配置 ε-递减策略的超参数，如 `start_epsilon`, `decay`, `min_epsilon`。

## 数据流

```mermaid
graph TD
    subgraph 训练循环
        A[调用 agent.act()] --> B{GreedyAlgorithm.run()}
    end

    subgraph GreedyAlgorithm
        B -- 根据 GreedyType --> C{选择手臂 (探索或利用)}
    end

    subgraph GreedyAgent
        A --> D[调用 env.pull() 获取奖励]
        D --> E[更新 Q-value 和计数]
    end

    style B fill:#cde4ff
    style C fill:#cde4ff
```

1.  **初始化**: 创建 `RLEnv`、`GreedyAlgorithm` 和 `GreedyAgent`。
2.  **决策**: 在训练循环中调用 `agent.act()`。
3.  **算法执行**: `act()` 内部调用 `algorithm.run()`。`GreedyAlgorithm` 根据其 `GreedyType`（例如 `EPSILON`）和当前状态（例如 `epsilon` 值）决定是探索还是利用，并返回一个手臂 `machine_id`。
4.  **环境交互与状态更新**: `GreedyAgent` 拉动 `machine_id` 对应的手臂，从环境中获得 `reward`，并用它来更新该手臂的 `q_value` 和 `count`。

## 用法示例

```python
from core.environment import RLEnv
from greedy.agent import GreedyAgent, GreedyAlgorithm
from greedy.schemas import GreedyType
from greedy.config import EpsilonDecreasingConfig

# 1. 初始化环境
env = RLEnv(machine_count=10, seed=42)

# 2. 初始化算法
# 使用 ε-递减贪婪算法，并启用乐观初始化
algorithm = GreedyAlgorithm(
    greedy_type=GreedyType.EPSILON_DECREASING,
    optimistic_init=True,
    optimistic_times=5
)

# 3. 初始化智能体
# 配置 ε 的起始值、衰减率和最小值
epsilon_config = EpsilonDecreasingConfig(
    start_epsilon=1.0, 
    decay=0.99, 
    min_epsilon=0.01
)

agent = GreedyAgent(
    name="EpsilonDecreasingGreedyAgent",
    env=env,
    algorithm=algorithm,
    epsilon_config=epsilon_config,
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
