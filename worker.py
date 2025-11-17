"""
worker.py (Actor 实现)

本文件实现单个 Actor 的循环（`actor_loop`），可以理解为“采样工人”：
- 周期性把 `global_net` 的参数拉取到本地 `local_net`（避免每次前向都跨进程访问）；
- 使用 `local_net` 在环境上采样 T_actor 步轨迹；
- 将轨迹打包并放入 `traj_queue` 供 Learner 消费。

关键数据格式（在 trajectory 中）：
- states: numpy array (T, C, L, L)
  - 每一步的棋盘状态，由 env.get_state() 产生；
- last_state: numpy array (C, L, L)
  - 用于 bootstrap 目标值 V(s_T)；
- actions: numpy array (T, L, L, 5)
  - 每格点上的概率向量（Dirichlet 采样结果，对应每个小组的分配比例）；
- behavior_log_probs: numpy (T,)
  - 每步行为策略的标量 log_prob（对每个 group 的 log_prob 做均值，变成标量）；
- rewards: numpy (T,)
  - 这里是 env 返回的 planner_reward（可以是累加式或 per-step）；
- entropies: numpy (T,)
  - 每步策略的平均熵，供 learner 做熵正则参考。

从“角色分工”角度看：
- Actor 负责“与环境互动 + 采样动作 + 记录数据”，不做参数更新；
- Learner 则离线拿到这些数据，集中更新 `global_net`，再同步回 Actor。
"""

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributions import Dirichlet
from env import PublicGoodsEnv
from planner_net import PlannerNet


def actor_loop(actor_id, global_net, traj_queue, device, L=32, r=1.4, T_actor=20):
    """
    单个 Actor 进程的主循环。

    主要步骤：
    1. 用 local_net 拉取 global_net 的参数
    2. 在 local_net + env 上采样 T_actor 步：
       - 把 local_net 的策略输出 alpha -> 构造 Dirichlet -> 采样得到每格点的概率向量 pi_field
       - 把 pi_field 传入 env.step(pi_field) -> 得到 next_state, reward, info
       - 收集 state, action(pi_field), behavior_log_prob, entropy, reward, done
    3. 把轨迹打包为 numpy/torch 可序列化结构并放入 traj_queue

    参数说明：
    - actor_id: int，供设置随机种子
    - global_net: 共享的全局网络（放在 shared memory）
    - traj_queue: multiprocessing.Queue，用于与 Learner 进程通信
    - device: torch device 字符串（通常 'cpu'）
    - L, r: 传给环境的网格大小与公共物品因子
    - T_actor: 每个 actor 采样的步数
    """
    torch.manual_seed(1234 + actor_id)
    np.random.seed(1234 + actor_id)

    env = PublicGoodsEnv(L=L, r=r)
    local_net = PlannerNet().to(device)

    state = env.get_state()

    while True:
        # 1. 同步参数（pull）
        local_net.load_state_dict(global_net.state_dict())

        states = []
        actions = []             # 每步保存 (L,L,5) 的 pi_field（Dirichlet 采样结果）
        behav_log_probs = []     # 每步行为策略的 scalar log_prob（对所有 group 取均值）
        rewards = []
        dones = []
        entropies = []

        for t in range(T_actor):
            # 把 numpy state 转成 tensor 送进 local_net
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1, C, L, L)

            with torch.no_grad():
                # local_net 输出 alpha（Dirichlet 的浓度参数）和 value（未使用于采样）
                alpha, value = local_net(s_tensor)         # alpha: (1,5,L,L)
                B, C5, H, W = alpha.shape                  # B=1, C5=5, H=W=L

                # 准备构造 N_groups 个 Dirichlet 分布（每个 group 一个），先 reshape
                alpha_flat = alpha.view(B, C5, -1).permute(0, 2, 1)  # (1, N_groups, 5)
                alpha_flat = alpha_flat[0]                            # (N_groups, 5)

                # 构造 batched Dirichlet：每个 group 一个参数 alpha
                dist = Dirichlet(alpha_flat)                          # N_groups 个 Dirichlet(5)

                # 采样动作：actions_flat 为 (N_groups, 5)，即每个 group 的概率向量
                actions_flat = dist.sample()                          # (N_groups, 5)

                # 计算行为策略的 log_prob 与熵，按 group 取平均得到每步的标量
                log_probs_flat = dist.log_prob(actions_flat)          # (N_groups,)
                behav_log_prob = log_probs_flat.mean()                # 标量

                entropy_flat = dist.entropy()                         # (N_groups,)
                entropy_t = entropy_flat.mean()                       # 标量

                # 恢复成 (L,L,5) 的 pi_field，直接传给 env
                pi_field = actions_flat.view(H, W, C5)                # (L,L,5)

            # env 真正执行 Dirichlet 采样出的 π_field
            pi_field_np = pi_field.cpu().numpy()
            next_state, reward, info = env.step(pi_field_np)

            # 记录轨迹数据
            states.append(state)
            actions.append(pi_field_np)                     # (L,L,5)
            behav_log_probs.append(behav_log_prob.cpu().item())
            rewards.append(float(reward))
            dones.append(False)                             # 这里暂不考虑 episode 终止
            entropies.append(entropy_t.cpu().item())

            state = next_state

        last_state = state

        # 打包轨迹，注意序列化友好类型（numpy arrays / primitive types）
        traj = {
            "states": np.stack(states, axis=0),            # (T, C, L, L)
            "last_state": last_state,                      # (C, L, L)
            "actions": np.stack(actions, axis=0),          # (T, L, L, 5)
            "behavior_log_probs": np.array(behav_log_probs, dtype=np.float32),  # (T,)
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "entropies": np.array(entropies, dtype=np.float32),                 # (T,)
        }

        traj_queue.put(traj)
