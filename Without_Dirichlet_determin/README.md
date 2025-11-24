TD3 Planner (Deterministic Actor)
=================================

本目录实现了一个 **确定性制度规划者 + TD3** 框架，用于在空间公共物品博弈环境中学习分配制度。

核心思想
--------

- 环境：`env.py` 中的 `PublicGoodsEnv`，与随机 Dirichlet 版本相同。
  - 状态 `state` 为 `(3, L, L)`：
    - 当前策略 Stra_now（合作者=1，背叛者=0）
    - 上一轮策略 Stra_prev
    - 以每个格点为中心时的小组公共池大小 P_center
  - 动作 `pi_field` 为 `(L, L, 5)`：每个格点输出一个 5 维分配比例（mid, up, down, left, right）。

- Actor（确定性）：
  - 文件：`planner_net.py` 中的 `ActorNet`
  - 输入：`state (B, 3, L, L)`
  - 输出：`pi (B, 5, L, L)`，每个格点 5 维经过 softmax 后是合法的概率向量。

- Critic（双 Q 网络）：
  - 文件：`planner_net.py` 中的 `CriticNet`
  - 输入：`state (B, 3, L, L)` 与 `action (B, 5, L, L)`
  - 输出：标量 `Q(s,a)`。

- TD3 训练逻辑：
  - 文件：`global_trainer.py` 中的 `train_td3` 函数；
  - 使用经验回放、双 Q 网络、target 网络、policy smoothing、延迟更新等标准 TD3 技巧；
  - Actor 是确定性的，探索通过在 logits 上加入高斯噪声实现。

文件结构
--------

- `env.py`  
  空间公共物品博弈环境，带 episode_length、reset、step 等接口。

- `planner_net.py`  
  - `ActorNet`：确定性制度规划者，CNN + 1x1 卷积输出每格点 5 维 logits，softmax 得到分配比例。  
  - `CriticNet`：单个 Q 网络，输入为 state 与 action 的通道拼接，输出标量 Q 值。

- `global_trainer.py`  
  TD3 训练入口：
  - 定义 `TD3Config` 超参数；
  - `ReplayBuffer` 经验回放；
  - `train_td3` 主循环：单环境、单进程。
  - 使用方式见下文。

如何运行
--------

在本目录下（`Without_Dirichlet_determin`）运行：

```bash
python global_trainer.py
```

或：

```bash
python -m global_trainer
```

默认配置（在 `__main__` 里，可自行修改）：

- 棋盘大小 `L = 25`，公共物品因子 `r = 4.0`
- 每个 episode 长度 `episode_length = 200`
- 总交互步数 `total_steps = 40000`
- 起始 `2000` 步采用纯探索（均匀分配），之后使用 Actor 输出 + 噪声探索

训练过程中，终端会打印：

- 当前总步数 step；
- 当前 episode 的长度、累积 reward；
- 当前 episode 的平均合作率 `mean_fC`。

训练结束后（且安装了 matplotlib），会在本目录下生成若干曲线图：

- `td3_episode_rewards.png`：每个 episode 的总奖励；
- `td3_episode_fC.png`：每个 episode 的平均合作率；
- `td3_losses.png`：critic loss 与 actor loss 随训练更新次数的变化。

超参数说明
----------

在 `TD3Config` 中可以调整：

- `gamma`：折扣因子；
- `actor_lr` / `critic_lr`：Actor / Critic 的学习率；
- `tau`：target 网络软更新系数（越小更新越平滑）；
- `policy_noise` / `noise_clip`：target smoothing 噪声大小与截断上限；
- `expl_noise`：行为策略在 logits 上的探索噪声；
- `policy_delay`：每多少次 critic 更新才更新一次 actor；
- `batch_size`：每次从 replay buffer 中采样的 batch 大小；
- `replay_size`：经验回放容量；
- `total_steps`：总环境交互步数；
- `start_steps`：前多少步只用随机策略探索。

后续扩展建议
------------

- 可以尝试增大或减小 `L`、`episode_length`，观察制度在不同尺度下的稳定性；
- 调整 `expl_noise` 与 `policy_noise`，控制策略的探索程度；
- 若希望与随机 Dirichlet/V-trace 版本对比，可复用同样的环境参数（L, r, episode_length），对比合作率与平均收益曲线。
