# 多臂老虎机项目

这个项目实现了多种解决多臂老虎机问题的算法，包括贪婪算法和UCB1算法。

## 项目结构

```
bandit/
├── core/              # 核心模块
│   ├── environment.py # 环境和老虎机类
│   ├── schemas.py     # 数据模型
│   └── README.md      # core模块说明
├── greedy/            # 贪婪算法模块
│   ├── algorithms.py  # 各种贪婪算法实现
│   ├── agent.py       # 贪婪算法代理
│   ├── config.py      # 算法配置
│   └── README.md      # greedy模块说明
├── ucb1/              # UCB1算法模块
│   ├── algorithms.py  # UCB1算法实现
│   └── README.md      # ucb1模块说明
├── train.py           # 训练函数
├── example_usage.py   # 使用示例
└── test_refactor.py   # 重构验证测试
```

## 模块说明

### Core 模块
包含项目的基础类和数据模型：
- `SlotMachine`：老虎机类
- `RLEnv`：强化学习环境类
- `RewardsState`：奖励状态模型
- `Metrics`：评估指标模型

### Greedy 模块
实现各种贪婪算法：
- 普通贪婪算法
- ε-贪婪算法
- ε-递减贪婪算法

### UCB1 模块
实现UCB1算法：
- 基于置信区间上界的算法实现

## 使用示例

可以运行 `example_usage.py` 文件查看如何使用这些模块：

```bash
python -m bandit.example_usage
```

## 测试

可以运行 `test_refactor.py` 文件验证重构是否成功：

```bash
python -m bandit.test_refactor