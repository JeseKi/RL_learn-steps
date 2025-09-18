"""TSAgent 模块测试

文件功能：
- 针对 `thompson_sampling.agent.TSAgent` 的公开接口进行单元测试，覆盖常规、边界和错误路径。

公开接口：
- pytest 测试函数：
  - `test_ts_agent_pull_machine_updates_state`
  - `test_ts_agent_pull_machine_invalid_machine_id_raises`

内部方法：
- 本测试文件不包含内部辅助方法。

公开接口的 pydantic 模型：
- 不适用（测试文件无对外返回模型）。
"""

from __future__ import annotations

import pytest

from core.environment import RLEnv
from thompson_sampling.agent import TSAgent, TSAlgorithm
from thompson_sampling.schemas import TSRewardsState


def _make_env_and_agent(machine_count: int = 5, seed: int = 123) -> TSAgent:
    """构造环境与 TSAgent（用于测试夹具）

    说明：
    - 使用固定 seed 以保证可重复性；
    - 仅通过 TSAgent 的公开构造函数创建实例。
    """
    env = RLEnv(machine_count=machine_count, seed=seed)
    algorithm = TSAlgorithm()
    agent = TSAgent(
        name="ts",
        env=env,
        algorithm=algorithm,
        convergence_threshold=0.9,
        convergence_min_steps=10,
        seed=seed,
    )
    return agent


def test_ts_agent_pull_machine_updates_state():
    """常规与边界：拉动一次机器后，counts 与 values 应按返回奖励正确更新

    说明：
    - 仅验证公开属性 `rewards.values` 与 `rewards.counts` 的变化；
    - alpha/beta 的具体更新策略不做强约束，仅断言其存在并执行了加一更新之一。
    """
    agent = _make_env_and_agent(machine_count=2, seed=1234)

    # 选择 0 号机器进行一次拉动
    machine_id = 0
    before_counts = list(agent.rewards.counts)
    before_values = list(agent.rewards.values)

    reward = agent.pull_machine(machine_id)

    # counts 应+1，values 应+reward
    assert agent.rewards.counts[machine_id] == before_counts[machine_id] + 1
    assert agent.rewards.values[machine_id] == before_values[machine_id] + reward

    # 若奖励为 1，则 alpha 应+1；若奖励为 0，则 beta 应+1
    assert isinstance(agent.rewards, TSRewardsState)
    a = agent.rewards.alpha[machine_id]
    b = agent.rewards.beta[machine_id]
    # 初始为 1，因此只会有其一从 1 变为 2
    assert (a == 2 and b == 1) or (a == 1 and b == 2)


def test_ts_agent_pull_machine_invalid_machine_id_raises():
    """错误路径：非法机器 ID 应触发断言异常

    依据：RLEnv.pull 对越界 machine_id 使用 assert 进行校验。
    """
    n = 3
    agent = _make_env_and_agent(machine_count=n, seed=99)

    with pytest.raises(ValueError, match="机器ID超出范围"):
        _ = agent.pull_machine(n)  # 越界：有效索引为 [0, n-1]
