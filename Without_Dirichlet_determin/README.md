TD3 Planner (Deterministic Actor)
=================================

本目录实现了一个 **确定性制度规划者 + TD3** 框架，用于在**空间公共物品博弈环境**中学习分配制度（planner）。  
这一版是「确定性 Actor + 双 Q 网络 + 经验回放 + target 网络 + policy smoothing + 延迟更新」的标准 TD3 结构，  
并针对本项目的公共物品环境做了适配（状态/奖励归一化、多网格参数评估等）。


核心组件概览
------------

- 环境：`env.py` 中的 `PublicGoodsEnv`
  - 棋盘：`L × L` 个体，每个体策略为合作者 C=1 或背叛者 D=0。
  - 小组：以任意格点 `(i,j)` 为中心，和上下左右共 5 个个体进行公共物品博弈。
  - 公共池：每个合作者贡献 `1`，总贡献乘以因子 `r` 得到公共池 `P`，再按 planner 给出的分配比例在 5 个个体之间分配。
  - 演化：个体根据邻居收益差异，通过 Fermi 规则更新策略。
  - 奖励：以「全局平均净收益」为基础，并做归一化处理，作为 planner 的 reward。

- 策略网络（Actor，确定性）：
  - 文件：`planner_net.py` 中的 `ActorNet`
  - 输入：`state (B, 4, L, L)`（新增累计资源通道）
  - 输出：`pi (B, 5, L, L)`，每个格点 5 维经过 softmax 后是合法的分配比例。

- 价值网络（Critic，双 Q）：
  - 文件：`planner_net.py` 中的 `CriticNet`
  - 输入：`state (B, 4, L, L)` 与 `action (B, 5, L, L)`
  - 输出：标量 `Q(s, a)`。

- 训练逻辑：
  - 文件：`global_trainer.py`
  - 使用 TD3 标准技巧：
    - 经验回放 `ReplayBuffer`
    - 双 Q 网络 + 取 min 减少过高估计
    - target 网络 + 软更新（Polyak averaging）
    - target policy smoothing（在 target actor logits 上加噪声并截断）
    - policy 延迟更新（多次 critic 更新后再更新一次 actor）
  - 额外特性：
    - 状态 / 奖励归一化（便于数值稳定、跨 `r` 对比）
    - 训练日志 CSV（包括 episode 指标和 eval 指标）
    - 支持自动保存 best 模型、可选早停
    - 单独的评估模式：在多个 `(L, r, episode_length)` 组合上评估同一个训练好的 planner。


环境 `PublicGoodsEnv` 说明
-------------------------

文件：`env.py`

- 状态 `state`：shape `(4, L, L)`，由 `get_state()` 构造：
  1. `Stra_now`：当前可合作策略（资源足且策略为 C 为 1，否则 0）
  2. `Stra_prev`：上一轮策略
  3. `P_center_norm`：以每个格点为中心小组的公共池大小 `P_center` 做归一化后的结果  
     - 原始公共池：`P_center[i,j] = r * n_c(i,j)`，其中 `n_c` 为该小组的合作者数（最多 5）  
     - 归一化：`P_center_norm = P_center / (5 * r)`，理论上落在 `[0, 1]` 左右，有利于网络训练
  4. `R_norm`：累计资源归一化，分母取“全局满合作时的稳态累计资源”  
     `denom ≈ 5*(r-1)/R_decay`，对应 `R_{t+1} = (1-R_decay)R_t + 5*(r-1)` 的稳态解

- 动作 `pi_field`：shape `(L, L, 5)`
  - 每个格点 `(i,j)` 对应一个 5 维向量：`[mid, up, down, left, right]`
  - 在环境中再次做一次归一化，以防数值偏差（确保和为 1）

- 奖励 `planner_reward`：
  - 先计算**未归一化**的平均净收益：
    - `total_P = sum_{groups} P_group = r * 总合作者出资次数`
    - `total_cooperators = #C`
    - `net_total = total_P - 5 * total_cooperators`（每个合作者参与 5 个小组）
    - `avg_net = net_total / (L^2)`（人均净收益）
  - 再按理论量级做归一化：
    - 归一化因子：`scale = 5 * (r - 1)`
    - `norm_avg_net = avg_net / scale`（大致落在一个相对稳定的区间）
  - 如果 `use_cumulative_planner_reward=False`（TD3 默认设置）：  
    每步 reward = `norm_avg_net`，适合 value-based / actor-critic 方法。
  - 如果 `use_cumulative_planner_reward=True`：  
    reward 会在 env 内部累加，用于「长期累计收益」的视角（本 TD3 脚本默认不使用）。

- 其他信息：
  - `info` 字典包含：
    - `avg_r`：本步平均即时收益
    - `avg_net`：未归一化的平均净收益
    - `f_C`：当前合作率
    - `avg_R`：累计资源平均值
    - `t` / `done`：当前步数 / 是否结束 episode


网络结构：`ActorNet` & `CriticNet`
---------------------------------

文件：`planner_net.py`

- `ConvBody`：
  - 共享的卷积骨干：3 层 `3×3` 卷积 + ReLU，不改变空间尺寸。
  - 用于抽取棋盘上的局部交互特征。

- `ActorNet`（确定性规划者）：
  - 输入：`state (B, 4, L, L)`
  - 结构：
    - `ConvBody(4, base_channels=32)`
    - `policy_head: 1×1 Conv` 输出 5 通道 logits
    - 前向中做 softmax 得到 `pi (B, 5, L, L)` 概率分布
  - 训练和行为时的探索并不在 `ActorNet` 内部实现，而是在 `global_trainer.py` 的 `select_action` 中对 logits 加高斯噪声。

- `CriticNet`（单个 Q 网络）：
  - 输入：`state (B, 4, L, L)` 和 `action (B, 5, L, L)`
  - 结构：
    - 在通道上 concat 得到 `(B, 8, L, L)`，喂入 `ConvBody(8, base_channels=32)`
    - 全局平均池化 + MLP 输出标量 `Q(s,a)`。
  - 训练时会同时维护两个 `CriticNet`：`critic1` 和 `critic2`，并配备各自的 target 网络。


训练与评估脚本：`global_trainer.py`
-----------------------------------

主要内容：

- `TD3Config`：集中管理所有 TD3 相关的超参数（见后文「超参数汇总」）。
- `ReplayBuffer`：经验回放缓冲区。
- `select_action`：给定 Actor 和状态，返回带探索噪声的 `pi_field`。
- `train_td3`：完整的 TD3 训练循环（支持多进程采样 + 单 learner）。
- `evaluate_trained_actor`：加载某个已训练 Actor，在新的 `(L, r, episode_length)` 组合上评估若干 episode。
- 多进程采样：配置 `rollout_workers` 可启动多个 CPU 进程并行生成样本，主进程用 GPU 训练；每个 worker 内部线程数限制为 1，避免占满所有 CPU。
- TensorBoard：训练/评估指标会写入 `runs_dir`（默认 checkpoints/<run_id>）：
  - 训练：`train/episode_reward`、`train/mean_fC`（来自 worker 的 episode 统计）
  - 评估：`eval/reward`、`eval/mean_fC`
  运行 `tensorboard --logdir checkpoints` 可实时查看曲线。
- `__main__`：
  - 构造一个 `TD3Config`；
  - 通过 `EVAL_ONLY` 开关控制是「训练」还是「只评估已有 run」。


运行方式
--------

### 1. 训练一个新的 TD3 planner

在 `Without_Dirichlet_determin` 目录下，将 `global_trainer.py` 末尾设置为：

```python
EVAL_ONLY = False
```

然后在该目录下运行：

```bash
python global_trainer.py
```

或：

```bash
python -m global_trainer
```

训练时会：

- 使用 `TD3Config` 中的超参数初始化 env / Actor / Critic / target 网络；
- 使用经验回放和 TD3 算法与环境交互、更新网络；若设置了 `rollout_workers>1`，会启多个 CPU 进程并行采样，主进程用 GPU 训练。
- 每个 episode 的统计（来自 worker 上报）会写入 `training_log.csv` 与 TensorBoard（train/episode_reward、train/mean_fC）。
- 每隔 `eval_interval` 步执行一次 eval（若 `eval_interval > 0`）：
  - 在一个全新的 env 上跑 `eval_episodes` 个 episode
  - 打印 eval 平均奖励和平均合作率
  - 写入 `training_log.csv` 与 TensorBoard（eval/reward、eval/mean_fC）
  - 若 eval reward 优于历史 best 且 `save_best=True`，会额外保存一份 `best_*.pt` 模型。

训练结束后：

- 在 `cfg.save_dir / <run_id>` 下保存：
  - `actor.pt`, `actor_target.pt`
  - `critic1.pt`, `critic2.pt`
  - `critic1_target.pt`, `critic2_target.pt`
  - 如开启 `save_best=True`：额外保存 `best_actor.pt` 等 best 模型
  - `training_log.csv`：包含 train episode 和 eval 的关键指标
  - 若安装了 matplotlib：
    - `td3_episode_rewards.png`：各 episode 奖励
    - `td3_episode_fC.png`：各 episode 合作率
    - `td3_losses.png`：actor / critic loss 曲线


### 2. 只评估已有 run（多参数扫描）

在 `global_trainer.py` 末尾，你可以配置：

```python
EVAL_ONLY = True
EVAL_RUN_ID = "20251119_230050第一版T3D较好效果"  # 要评估的 run_id（对应 checkpoints 子目录名）
EVAL_L_LIST = [25, 30, 35, 40]
EVAL_R_LIST = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
EVAL_EPISODE_LENGTH_LIST = [150]
EVAL_EPISODES = 10
```

含义：

- 会从 `os.path.join(cfg.save_dir, EVAL_RUN_ID, "actor.pt")` 加载 Actor；
- 对给定的所有 `(L, r, episode_length)` 组合逐一评估：
  - 每个组合跑 `EVAL_EPISODES` 个 episode；
  - 评估时不加探索噪声（完全按策略执行）；
  - 记录每个组合的平均 reward 和平均合作率。

运行：

```bash
python global_trainer.py
```

会生成一个 CSV：

- 名字类似：`eval_results_<EVAL_RUN_ID>_<timestamp>.csv`
- 列包括：`L, r, episode_length, eval_episodes, mean_reward, mean_fC`

这就是你现在已有的那几份 `eval_results_*.csv` 文件的来源。


默认配置（当前代码）
--------------------

在 `__main__` 中，当前示例配置为：

- TD3 超参数（`TD3Config`）：
  - `device = "cuda:0"`（若无 GPU 可改为 `"cpu"`）
  - `gamma = 0.99`
  - `actor_lr = 1e-4`
  - `critic_lr = 1e-4`
  - `tau = 0.005`
  - `policy_noise = 0.10`
  - `noise_clip = 0.20`
  - `expl_noise = 0.0`（行为策略阶段不额外加噪声）
  - `policy_delay = 2`
  - `batch_size = 32`
  - `replay_size = 100_000`
  - `total_steps = 400_000`
  - `start_steps = 20_000`（前 2 万步使用均匀分配作为纯探索）
  - `eval_interval = 5_000`
  - `eval_episodes = 3`
  - `save_models = True`
  - `save_dir = <当前目录>/checkpoints`
  - `load_run_id = None`（不从旧 run 续训）
  - `save_best = True`
  - `early_stop_patience = 0`（不启用早停）

- 环境参数（训练模式下）：
  - `L = 25`
  - `r = 4.0`
  - `episode_length = 150`
  - `use_cumulative_planner_reward=False`（使用 per-step 归一化平均净收益）

你可以根据需要修改这些默认值，然后重新运行脚本。


超参数汇总与调整建议
--------------------

`TD3Config` 中主要超参数及建议：

- `gamma`：折扣因子
  - 一般保持在 `0.95 ~ 0.99`，越接近 1 越看重长期效果。

- `actor_lr`, `critic_lr`：学习率
  - 目前都设为 `1e-4`，如训练不稳定可适当减小。

- `tau`：target 网络软更新系数
  - 值越小，target 更新越慢，训练更平滑但收敛更慢；常用 `0.005 ~ 0.02`。

- `policy_noise`, `noise_clip`：
  - 控制 target policy smoothing 的强度，防止 Q 函数在动作空间局部极值处过拟合。
  - 一般 `policy_noise` 取 `0.1` 左右，`noise_clip` 取 `2×policy_noise`。

- `expl_noise`：
  - 行为策略在 logits 上的探索噪声，目前示例配置设为 `0.0`，即仅靠环境随机性 / Fermi 演化产生多样性。
  - 如果希望 Actor 行为更具随机性，可设置为 `0.05 ~ 0.2` 之间。

- `policy_delay`：
  - TD3 中通常取 2，即每 2 次 critic 更新，更新一次 actor。

- `batch_size`, `replay_size`：
  - batch 越大估计越稳定但单次更新开销更大；
  - replay_size 较大可以提高经验多样性，当前 `1e5` 对本环境比较合理。

- `total_steps`, `start_steps`：
  - `start_steps` 控制纯探索阶段长度（不依赖 Actor），可以视为「先收集多少均匀分配的数据」。
  - 对复杂任务适当增大 `start_steps` 通常有利于稳定训练（但会延长总时间）。

- `eval_interval`, `eval_episodes`：
  - 决定多频繁做一次 eval 以及每次 eval 的统计精度。
  - 如果计算资源紧张，可以降低 eval 频率或者减少 `eval_episodes`。

- `save_models`, `save_best`, `early_stop_patience`：
  - 若只想快速试验，不在乎中间结果，可暂时关掉 `save_models` 或缩小 `replay_size`、`total_steps`。
  - 当 `early_stop_patience > 0` 且多次 eval reward 未提升时会提前结束训练。


对比随机 Dirichlet 版本
------------------------

本目录是确定性 TD3 版本，对应的随机 Dirichlet + V-trace 版本在上一级目录的 `global_trainer.py` 中。

- 随机版本：
  - 策略是 Dirichlet（每步在每个格点采样一个概率向量），训练使用 V-trace / A3C 风格的 on/off-policy policy gradient。
- 本 TD3 版本：
  - 策略是确定性的概率场 `pi_field`，训练使用 TD3（value-based + deterministic policy gradient）。

可以在相同的 `(L, r, episode_length)` 下，比对两种方法的：

- 收敛速度；
- 最终平均净收益；
- 合作率轨迹 `f_C` 的演化形态。  

这也是本仓库设计两套实现的主要目的之一。
