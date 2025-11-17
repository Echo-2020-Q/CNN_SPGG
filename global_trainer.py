"""
global_trainer.py (Learner + launcher)

这个模块实现 Learner（主进程）和训练入口，是“整个系统的启动器 + 训练大脑”。

总体架构（从宏观到微观）：
- environment: `PublicGoodsEnv`
  - 定义公共物品博弈规则和 Fermi 策略演化规则；
- policy/value network: `PlannerNet`
  - 把 env.get_state() 的 5 通道状态映射成 Dirichlet 参数 alpha 和 value；
- actors (在 worker.py 中实现):
  - 多个进程并行运行环境 + 策略，采集轨迹；
  - 每个 actor 把 T_actor 步轨迹打包放入 `traj_queue`；
- learner (本文件中的 learner_loop):
  - 从队列中读取轨迹，用当前 global_net 对整条轨迹前向；
  - 用 V-trace 做 off-policy 校正，计算 policy_loss + value_loss；
  - 反向传播更新 global_net 的参数。

实现细节与注意点：
- 使用 Dirichlet 分布作为策略（行为是每个格点的概率向量），因此 `PlannerNet` 的策略头输出被解释为 Dirichlet 的 alpha（浓度参数）。
- trajectory 中 `actions` 的格式为 (T, L, L, 5)，表示每步每个格点的概率向量（由 Actor 的 Dirichlet 采样得到）。
- learner 会把这些 action 与当前策略下的 Dirichlet 分布做 log_prob 比较，从而得到 importance weight（用于 V-trace）。

如何运行（最简单方式）：
- 在本目录执行 `python global_trainer.py` 会启动若干 actor 进程并在主进程运行 learner_loop。
  可以先把 num_actors, L, max_updates 调小，用于本地验证流程是否正常。

调试建议：
- 多进程调试时请先用较小的 `num_actors`、`T_actor`、`max_updates` 进行本地测试；
- 经常打印 info 中的合作率 f_C、平均资源等，帮助理解学习是否朝着“更合作”的方向发展。
"""

import torch
import torch.multiprocessing as mp
import numpy as np
from torch.distributions import Dirichlet
from planner_net import PlannerNet
from worker import actor_loop
from worker_utils import compute_vtrace_loss
from env import PublicGoodsEnv   # 仅用于 L 参数


def learner_loop(
    global_net,
    traj_queue,
    device="cpu",
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
    entropy_beta=0.01,
    lr=1e-4,
    max_updates=1000,
):
    """
    Learner 主循环：
    - 从 traj_queue 获取单条轨迹（由任意 Actor 放入）
    - 把轨迹的 states 拼成一个大 batch (T+1, C, L, L)，对 global_net 做一次前向
    - 使用 global_net 的分布参数与轨迹中的 actions 计算 target_log_probs
    - 使用 V-trace 计算 loss, backward, optimizer.step()

    参数说明：参见调用处与默认值
    """
    optimizer = torch.optim.Adam(global_net.parameters(), lr=lr)

    for update_idx in range(max_updates):
        batch = traj_queue.get()

        # 从轨迹中解析变量
        states = batch["states"]             # numpy (T, C, L, L)
        last_state = batch["last_state"]     # numpy (C, L, L)
        actions_np = batch["actions"]        # numpy (T, L, L, 5)
        rewards = batch["rewards"]           # numpy (T,)
        dones = batch["dones"]               # numpy (T,)
        behav_log_probs_np = batch["behavior_log_probs"]  # numpy (T,)

        T, C, L, W = states.shape

        # 转成 tensor 并转到 device
        states_tensor = torch.from_numpy(states).float().to(device)          # (T,C,L,L)
        last_state_tensor = torch.from_numpy(last_state).float().unsqueeze(0).to(device) # (1,C,L,L)
        actions_tensor = torch.from_numpy(actions_np).float().to(device)     # (T,L,L,5)

        behav_log_probs = torch.from_numpy(behav_log_probs_np).to(device)    # (T,)
        rewards_t = torch.from_numpy(rewards).to(device)                     # (T,)
        dones_t = torch.from_numpy(dones.astype(np.float32)).to(device)      # (T,)

        # 把 states 与 last_state 合并，得到 (T+1, C, L, L)
        all_states = torch.cat([states_tensor, last_state_tensor], dim=0)    # (T+1,C,L,L)

        # 前向 global_net，得到 alpha_all (T+1,5,L,L) 与 values_all (T+1,)
        alpha_all, values_all = global_net(all_states)
        # 只取前 T 个 time step 的 alpha 来构建目标策略分布
        alpha_t = alpha_all[:-1]                                             # (T,5,L,L)

        # reshape alpha_t, actions 成 (T, N_groups, 5)
        B = T
        _, C5, H, W2 = alpha_t.shape
        assert H == L and W2 == W

        alpha_flat = alpha_t.view(B, C5, -1).permute(0, 2, 1)               # (T,N_groups,5)
        actions_flat = actions_tensor.view(B, -1, C5)                       # (T,N_groups,5)

        # 构造 batched Dirichlet（每个 group 一个 Dirichlet）并计算 log_prob
        dist_target = Dirichlet(alpha_flat)
        target_log_probs_per_group = dist_target.log_prob(actions_flat)      # (T,N_groups)
        # 这里对所有 group 取平均得到每步的 scalar log_prob（便于 V-trace 的单变量实现）
        target_log_probs = target_log_probs_per_group.mean(dim=1)            # (T,)

        # 用当前策略的熵做正则（也可以使用行为方差）
        entropy_per_group = dist_target.entropy()                            # (T,N_groups)
        entropy_t = entropy_per_group.mean(dim=1)                            # (T,)

        # values_all 已经是 (T+1,)
        values = values_all                                                  # (T+1,)

        # 计算 V-trace 损失并更新参数
        loss, pl, vl = compute_vtrace_loss(
            target_log_probs=target_log_probs,
            behavior_log_probs=behav_log_probs,
            values=values,
            rewards=rewards_t,
            dones=dones_t,
            entropies=entropy_t,
            gamma=gamma,
            rho_bar=rho_bar,
            c_bar=c_bar,
            entropy_beta=entropy_beta,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(global_net.parameters(), 40.0)
        optimizer.step()

        if update_idx % 10 == 0:
            print(f"[Learner] update {update_idx}, loss={loss.item():.4f}, "
                  f"policy_loss={pl:.4f}, value_loss={vl:.4f}")


def train(
    num_actors=4,
    L=32,
    r=1.4,
    device="cpu",
    max_updates=1000,
):
    """
    训练入口：
    - 创建共享 global_net (放到 shared memory)
    - 创建 traj_queue
    - 启动若干 Actor 进程（每个运行 actor_loop）
    - 在主进程运行 learner_loop
    """
    global_net = PlannerNet().to(device)
    global_net.share_memory()  # 共享参数

    traj_queue = mp.Queue(maxsize=64)

    # 启动多个 Actor
    actors = []
    for i in range(num_actors):
        p = mp.Process(
            target=actor_loop,
            args=(i, global_net, traj_queue, device, L, r, 20),  # T_actor=20
        )
        p.daemon = True
        p.start()
        actors.append(p)

    # Learner 在主进程里跑
    learner_loop(
        global_net=global_net,
        traj_queue=traj_queue,
        device=device,
        max_updates=max_updates,
    )

    for p in actors:
        p.terminate()
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    train(num_actors=4, L=16, r=1.4, device="cpu", max_updates=200)
