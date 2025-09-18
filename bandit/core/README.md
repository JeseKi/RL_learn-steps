# Core 模块

`core` 模块提供了多臂老虎机（Multi-Armed Bandit）问题模拟所需的核心抽象基类和基础组件。它将问题分解为 `Environment`、`Agent` 和 `Algorithm` 三个主要部分，为实现和评估不同的 Bandit 算法提供了统一的框架。

## 业务逻辑

该模块旨在将 Bandit 问题的环境、智能体（决策者）和算法（决策逻辑）进行解耦。

-   **环境（Environment）**: `RLEnv` 类负责模拟外部环境。它包含一组“老虎机”（`SlotMachine`），每个都有不同的奖励概率。它还支持非平稳（non-stationary）环境，即奖励概率可以随着时间的推移而发生变化（例如随机游走或分段变化）。
-   **智能体（Agent）**: `BaseAgent` 是所有智能体的抽象基类。它负责维护与环境交互的状态（例如，每个手臂的预估奖励值、被选择的次数），并执行与环境的交互（`pull_machine`）。此外，它还负责计算关键的性能指标，如 `regret`（后悔值）和 `optimal_rate`（最优臂选择率）。
-   **算法（Algorithm）**: `BaseAlgorithm` 是所有决策算法的抽象基类。它封装了选择手臂的具体逻辑（例如，Epsilon-Greedy、UCB1 等）。`Agent` 会持有一个 `Algorithm` 实例，并在每次决策时调用它。

## 公开接口

### `core.environment.RLEnv`

强化学习环境类。

-   `__init__(...)`: 初始化环境，可配置手臂数量、是否为非平稳环境等。
-   `pull(machine_id: int, steps: int) -> int`: 拉动指定 `machine_id` 的手臂，并返回奖励（0 或 1）。
-   `best_reward(steps: int) -> float`: 计算在 `steps` 步内可能获得的最大累积奖励。

### `core.schemas.BaseAgent`

智能体抽象基类。

-   `act(**kwargs) -> int`: 执行一次决策，选择一个手臂并与环境交互，然后更新内部状态。
-   `pull_machine(machine_id: int) -> int`: 拉动指定的手臂，获取奖励并更新状态。
-   `metric() -> Metrics`: 返回当前步数的各项性能指标，包括后悔值、后悔率和最优臂选择率。

### `core.schemas.BaseAlgorithm`

算法抽象基类。

-   `run() -> int`: 根据具体算法逻辑，选择并返回一个手臂的索引。

### `core.schemas.Metrics`

用于封装性能指标的 Pydantic 模型，包含 `regret`, `regret_rate`, `rewards`, `optimal_rate`。

## 数据流

典型的训练和评估流程如下：

1.  **初始化**:
    -   创建一个 `RLEnv` 实例，定义好多臂老虎机环境。
    -   创建一个具体的 `Agent` 实例（例如，`GreedyAgent`），并为其注入一个具体的 `Algorithm` 实例（例如，`EpsilonGreedyAlgorithm`）和 `RLEnv` 实例。

2.  **训练循环**:
    -   在一个固定步数（`T`）的循环中，重复调用 `agent.act()` 方法。
    -   在 `act()` 方法内部：
        a.  `Agent` 调用其 `algorithm.run()` 方法来获取决策（即选择哪个手臂）。
        b.  `Agent` 调用 `pull_machine()` 方法，该方法会调用 `env.pull()` 与环境交互，获得奖励。
        c.  `Agent` 根据获得的奖励更新其内部状态（例如，更新 Q-value 估算）。
        d.  （可选）在每一步或特定间隔，调用 `agent.metric()` 来记录当前的性能指标。

3.  **评估**:
    -   训练循环结束后，通过 `Agent` 中记录的历史性能指标来分析和可视化算法的表现。

## 用法示例

```python
from core.environment import RLEnv
from core.schemas import BaseAgent, BaseAlgorithm

# 1. 定义具体的 Algorithm
class MyGreedyAlgorithm(BaseAlgorithm):
    def run(self) -> int:
        # 在 Agent 中实现具体的逻辑
        if self.agent.rng.random() < 0.1:  # 10% 的概率探索
            return self.agent.rng.integers(0, len(self.agent.rewards.values))
        else: # 90% 的概率利用
            return np.argmax(self.agent.rewards.values)

# 2. 定义具体的 Agent
class MyAgent(BaseAgent):
    def act(self) -> int:
        machine_id = self.algorithm.run()
        self.pull_machine(machine_id)
        self.steps += 1
        return machine_id

    def pull_machine(self, machine_id: int) -> int:
        reward = self.env.pull(machine_id, self.steps)
        # 更新状态
        self.rewards.counts[machine_id] += 1
        self.rewards.values[machine_id] += reward
        return reward

# 3. 运行模拟
# 初始化环境
env = RLEnv(machine_count=10)
# 初始化 Agent 和 Algorithm
agent = MyAgent(name="MyGreedyAgent", env=env, algorithm=MyGreedyAlgorithm(...))

# 训练循环
for _ in range(1000):
    agent.act()

# 打印最终的后悔值
print(f"Total Regret: {agent.regret()}")
```
