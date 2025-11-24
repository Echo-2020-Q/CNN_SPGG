++ Project2/README.md
"""
Project2 — Planner + Actor-Learner framework for Spatial Public Goods Game

目录结构（本文件夹）
- `env.py`         : 环境实现 `PublicGoodsEnv`（空间公共物品博弈）
- `planner_net.py` : 卷积式策略-价值网络 `PlannerNet`（输出 Dirichlet alpha + value）
- `worker.py`      : Actor 实现（`actor_loop`），并行采样轨迹
- `global_trainer.py`: Learner + 训练入口（启动 actors、运行 learner_loop）
- `worker_utils.py`: 损失工具，包括 V-trace 与 A3C 损失实现

总体说明
-------------
该项目实现了一个基于 actor-learner 的并行强化学习框架（类似 IMPALA）：
1. 多个 Actor 进程并行和 `PublicGoodsEnv` 交互，使用本地网络采样动作（这里动作是每个格点的概率向量，使用 Dirichlet 采样）。
2. Actor 把采集到的轨迹打包到 `mp.Queue`（`traj_queue`），由 Learner（主进程）消费。
3. Learner 使用全局网络对轨迹做前向计算，重构 target_log_probs 与 values，并用 V-trace 算法计算损失更新 global_net。

关键实现细节
----------------
- 策略参数化：PlannerNet 输出 alpha（>0），作为 Dirichlet 的浓度参数；Actor 从 Dirichlet 中采样得到每格点的分配向量（pi_field），并把 pi_field 直接用于 `env.step`。
- 奖励设计：`env.py` 中实现了可选的累加式 planner reward（`use_cumulative_planner_reward`）与累计资源衰减 `R_decay`。
- V-trace：在 `worker_utils.py` 中实现，Learner 使用 target_log_probs（基于当前策略）与 trajectory 中的 behavior_log_probs 计算重要性比率并修正 value/advantage。

如何运行（本地快速调试）
---------------------------
1) 先在命令行用小规模参数运行，便于排错：

```bash
python Project2/global_trainer.py
```

你可以在 `global_trainer.py` 中把 `train()` 的参数改小：例如 `num_actors=1`, `max_updates=20`, `T_actor=5`。

2) 单步调试 env：

```bash
python - <<'PY'
from Project2.env import PublicGoodsEnv
import numpy as np
env = PublicGoodsEnv(L=8, r=1.4, R_decay=0.1, use_cumulative_planner_reward=True)
pi_field = np.ones((8,8,5), dtype=float)
pi_field = pi_field / pi_field.sum(axis=-1, keepdims=True)
for t in range(5):
    s, reward, info = env.step(pi_field)
    print(t, reward, info)
PY
```

调试建议
---------
- 首先确认 Actor 采样动作确实影响 env（当前实现即为如此：Actor 用 Dirichlet 采样得到的 pi_field 直接传给 env.step）。
- 确认 `behavior_log_probs` 与 Learner 中重构的 `target_log_probs` 使用相同的动作编码方式，例如都把 per-group log_prob 做平均来得到每步的 scalar log_prob。
- 若训练发散或不收敛：试试调整 reward scale（归一化）、降低学习率、关闭累计式 reward（`use_cumulative_planner_reward=False`）或降低 `alpha` 的初始尺度。

文件注释
---------
我已在每个源码文件中加入了大量注释，解释数据形状、参数含义与实现逻辑。建议从 `env.py` → `planner_net.py` → `worker.py` → `global_trainer.py` → `worker_utils.py` 的顺序阅读，以便逐步构建理解。

如果你希望，我可以：
- 把项目改成只用 A3C（单进程多线程）以便调试；或
- 把 Actor 的行为 log_prob 从 "group 平均" 改为 "group 求和"（或反之），并在 Learner 中与之保持一致；或
- 添加一个小脚本 `sanity_run.py`，用最小配置运行 1 个 actor+learner 并打印若干中间变量便于观察。

"""
