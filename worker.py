"""
worker.py (Actor 实现)

本文件实现单个 Actor 的循环（`actor_loop`），可以理解为“采样工人”：
- 周期性把 `global_net` 的参数拉取到本地 `local_net`（避免每次前向都跨进程访问）；
- 使用 `local_net` 在环境上采样 T_actor 步轨迹；
- 将轨迹打包并放入 `traj_queue` 供 Learner 消费。

关键数据格式（在 trajectory 中）：
- states: numpy array (T, C, L, L)
  - 每一步的棋盘状态，由 env.get_state() 产生；当前 C=3，对应 [Stra_now, Stra_prev, P_center]
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


def actor_loop(actor_id, global_net, traj_queue, device, L=32, r=1.4, T_actor=20, episode_length=500):
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

    env = PublicGoodsEnv(L=L, r=r, episode_length=episode_length)
    local_net = PlannerNet().to(device)

    #从环境中获取这一步的状态
    state = env.get_state()

    while True:
        # 1. 同步参数（pull）
        local_net.load_state_dict(global_net.state_dict())

        # 下面这些列表用于在当前采样周期内，按时间顺序缓存每一步的轨迹数据
        states = []              # 每一步的环境状态 state，形状为 (C, L, L)
        actions = []             # 每步保存 (L, L, 5) 的 pi_field（Dirichlet 采样结果）
        behav_log_probs = []     # 每步行为策略的标量 log_prob（对所有 group 的 log_prob 取均值）
        rewards = []             # 每步环境返回给 planner 的奖励（标量）
        dones = []               # 每步的终止标记，来自 env.step 的 done（episode 是否结束）
        entropies = []           # 每步策略的平均熵（对所有 group 的熵取均值）
        f_cs = []                # 每步的合作率 f_C（从 env.step 返回的 info 中读取）

        done = False             # 标记当前轨迹内是否在某一步结束了 episode
        for t in range(T_actor):
            # 把 numpy state 转成 tensor 送进 local_net
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1, C, L, L)

            with torch.no_grad():
                # local_net 输出 alpha（Dirichlet 的浓度参数）和 value（未使用于采样）
                alpha, value = local_net(s_tensor)         # alpha: (1,5,L,L)
                B, C, H, W = alpha.shape                   # B=1, C=5, H=W=L

                # 准备构造 N_groups 个 Dirichlet 分布（每个 group 一个），先 reshape
                alpha_flat = alpha.view(B, C, -1).permute(0, 2, 1)   # (1, N_groups, 5)
                alpha_flat = alpha_flat[0]                           # (N_groups, 5)

                # 构造 batched Dirichlet：每个 group 一个参数 alpha ; dist=distribution（分布）
                dist = Dirichlet(alpha_flat)                          # N_groups 个 Dirichlet(5)

                # 采样动作：actions_flat 为 (N_groups, 5)，即每个 group 的概率向量
                actions_flat = dist.sample()                          # (N_groups, 5)

                # 这一段是在把“每个格点上的信息”压缩成“当前这一步的两个标量”：
                #行为策略在这一步的平均 log_prob（标量）
                #行为策略在这一步的平均熵（标量）
                log_probs_flat = dist.log_prob(actions_flat)          # (N_groups,)
                # dist 里有 N_groups = L*L 个 Dirichlet 分布（每个格点一个）。
                # actions_flat 是 (N_groups, 5)，每行是该格点采样到的 5 维概率向量。
                # log_prob 对每个格点算 log π(a_t^group | s_t)，得到一个长度为 N_groups 的向量。

                behav_log_prob = log_probs_flat.mean()                # 标量
                # 对所有格点的 log_prob 求平均，变成一个标量。
                # 这个标量就代表“在这一步 t，整张棋盘的平均行为 log_prob”。
                # 存到轨迹里时，用这个标量就够了，Learner 的 V-trace 也是按标量来算的。

                #对每个格点的 Dirichlet 分布算熵，得到每个格点的“策略不确定性”。
                #再对所有格点求平均，得到这一步 t 的“平均策略熵”标量 entropy_t。
                #这个标量后面会用在熵正则（鼓励策略不要太确定，多探索一些）。
                entropy_flat = dist.entropy()                         # (N_groups,)
                entropy_t = entropy_flat.mean()                       # 标量

                # 恢复成 (L,L,5) 的 pi_field，直接传给 env
                pi_field = actions_flat.view(H, W, C)                 # (L,L,5)

            # env 真正执行 Dirichlet 采样出的 π_field
            pi_field_np = pi_field.cpu().numpy()
            next_state, reward, done, info = env.step(pi_field_np)

            # 记录轨迹数据
            states.append(state)
            actions.append(pi_field_np)                     # (L,L,5)
            behav_log_probs.append(behav_log_prob.cpu().item())
            rewards.append(float(reward))
            dones.append(bool(done))                        # 记录当前步是否 episode 结束
            entropies.append(entropy_t.cpu().item())
            f_cs.append(float(info.get("f_C", 0.0)))        # 记录当前步的合作率 f_C

            # 更新下一步要用的状态
            state = next_state

            # 如本步 episode 结束，则提前截断轨迹
            if done:
                break

        # 轨迹最后一个状态（用于 bootstrap），就是循环结束时的 state
        last_state = state

        # 如果 episode 已经结束，则在开始下一条轨迹前重置环境
        if done:
            state = env.reset()

        # 打包轨迹，注意使用 numpy / 基本类型，方便通过多进程队列传输与在 Learner 端处理
        traj = {
            "states": np.stack(states, axis=0),            # (T, C, L, L)  每步的环境状态序列
            "last_state": last_state,                      # (C, L, L)    轨迹最后一步的状态，用于 bootstrap V(s_T)
            "actions": np.stack(actions, axis=0),          # (T, L, L, 5) 每步每个格点的 Dirichlet 采样结果 pi_field
            "behavior_log_probs": np.array(behav_log_probs, dtype=np.float32),  # (T,)  每步行为策略的标量 log_prob
            "rewards": np.array(rewards, dtype=np.float32),                     # (T,)  每步的 reward 序列
            "dones": np.array(dones, dtype=bool),                               # (T,)  每步是否终止（True 表示 episode 结束）
            "entropies": np.array(entropies, dtype=np.float32),                 # (T,)  每步策略平均熵的时间序列
            "f_Cs": np.array(f_cs, dtype=np.float32),                           # (T,)  每步合作率的时间序列
        }

        traj_queue.put(traj)
