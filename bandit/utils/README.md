# Utils 模块

`utils` 模块提供了一系列用于实验数据记录、处理和可视化的工具。它不包含任何核心的 Bandit 算法逻辑，而是作为整个实验框架的辅助支持系统。

## 业务逻辑

该模块的功能主要分为两大块：数据保存和数据绘图。

### 1. 数据保存 (`save_data.py`)

为了能够详细分析算法在训练过程中的表现，我们需要记录其关键指标随时间步的变化。`ProcessDataLogger` 类为此而设计。

-   **对数采样**: 在长时间的训练（如百万步）中，如果每一步都记录数据，文件会变得异常庞大。该 Logger 使用对数网格（Logarithmic Grid）进行采样，即在训练早期密集采样，在后期稀疏采样。这既能捕捉到初期的快速变化，又能有效控制数据量。
-   **过程数据**: 记录的数据是“过程性”的，包含每个采样点的详细状态（如后悔值、Q-value、最优臂选择率等）。

### 2. 数据绘图 (`plot_*.py`)

模块的核心功能是将 `ProcessDataLogger` 保存的 `*process.json` 文件可视化，以便于算法性能的比较和分析。

-   **数据聚合 (`plot_aggregate.py`)**: 在进行多次独立实验（例如，使用不同的随机种子）后，绘图工具首先会按算法名称对所有实验运行进行分组，然后计算每个算法在所有运行中的“平均性能”。它会找到所有运行的公共时间步，并对指标进行均值聚合，从而生成平滑的平均性能曲线。
-   **2x2 对比图 (`plot_cli.py`)**: 这是最主要的绘图功能。它会创建一个 2x2 的图表，同时展示四种核心指标（如累积奖励、后悔值、后悔率、最优臂选择率）在不同算法下的对比情况。
-   **智能标注 (`plot_intersections.py`)**: 为了增强图表的可读性，该模块能自动检测并标注：
    -   不同算法性能曲线之间的**交点**。
    -   曲线与坐标轴的交点。
    -   算法达到**收敛阈值**（如最优臂选择率达到 90%）的步数。
-   **命令行接口 (`plot_cli.py`)**: 提供了一个方便的命令行工具，用户可以直接通过终端调用绘图功能，而无需编写额外的 Python 脚本。

## 公开接口

### `utils.save_data.ProcessDataLogger`

用于在训练过程中记录详细数据的核心类。

-   `__init__(run_id, total_steps, grid_size)`: 初始化 Logger，并根据总步数和网格大小创建对数采样点。
-   `should_record(step)`: 判断当前步数是否需要记录。
-   `add(step, data)`: 添加一条数据记录。
-   `save(file_name, total_steps)`: 将所有记录的数据保存到指定的 JSON 文件中。

### `utils.plot.cli_main`

命令行绘图工具的入口函数。可以通过 `python -m utils.plot_cli` 来调用。

-   它接受一个或多个 `*process.json` 文件作为输入。
-   支持自定义图表标题和输出路径。
-   能够将多个实验运行的数据聚合在一起进行对比。

## 数据流

一个典型的“训练-分析”流程如下：

```mermaid
graph TD
    subgraph 训练阶段 (train.py)
        A[开始训练循环] --> B{ProcessDataLogger.should_record?}
        B -- Yes --> C[记录当前步的指标]
        B -- No --> A
        C --> A
        A -- 循环结束 --> D[Logger.save("...")]
    end

    subgraph 分析阶段 (CLI)
        E[用户执行 python -m utils.plot_cli *.json] --> F[加载并聚合数据]
        F --> G[绘制 2x2 对比图]
        G --> H[自动标注交点/收敛点]
        H --> I[保存为 PNG 图片]
    end

    D -.-> E
```

1.  **训练时**: 在主训练脚本中，创建一个 `ProcessDataLogger` 实例。在每个训练步，使用 `logger.should_record()` 判断是否需要记录，如果需要，则调用 `logger.add()` 保存当前状态。训练结束后，调用 `logger.save()` 将数据写入 `*process.json` 文件。
2.  **分析时**: 在终端中，运行 `python -m utils.plot_cli` 命令，并传入一个或多个之前生成的 `*process.json` 文件。
3.  **绘图工具**会自动完成数据加载、分组、聚合、绘图和智能标注的所有工作，并最终生成一张包含详细对比信息和标注的 PNG 图表。

## 用法示例

**命令行绘图:**

```bash
# 比较两个不同算法的实验结果
python -m utils.plot_cli experiment_data/greedy_run.json experiment_data/ucb1_run.json --title "Greedy vs. UCB1" -o reports/greedy_vs_ucb1.png

# 聚合一个算法的多次运行结果
python -m utils.plot_cli experiment_data/ts_run_*.json --title "Thompson Sampling (Avg of 5 runs)" -o reports/ts_avg.png
```
