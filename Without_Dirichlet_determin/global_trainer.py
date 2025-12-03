"""
TD3 trainer for the deterministic planner in the spatial public-goods game.

单进程 + 向量化环境版本：使用 BatchedPublicGoodsEnv 在一个进程内并行推进多个棋盘，
避免多进程 IPC 开销。

用法：
    python global_trainer.py
"""

from __future__ import annotations

import os
import datetime
import random
import csv
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.tensorboard import SummaryWriter

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from env import PublicGoodsEnv, BatchedPublicGoodsEnv
from planner_net import ActorNet, CriticNet

STATE_CHANNELS = 4  # stra_now, stra_prev, P_center_norm, R_norm


@dataclass
class TD3Config:
    device: str = "cpu"                      # 训练设备（如 "cpu" 或 "cuda:0"）
    gamma: float = 0.99                      # 折扣因子
    actor_lr: float = 1e-4                   # Actor 学习率
    critic_lr: float = 1e-4                  # Critic 学习率
    tau: float = 0.005                       # target 网络软更新系数
    policy_noise: float = 0.1                # target policy smoothing 的噪声强度
    noise_clip: float = 0.2                  # target smoothing 噪声截断范围
    expl_noise: float = 0.0                  # 行为策略探索噪声（加在 logits 上）
    policy_delay: int = 2                    # 每多少次 critic 更新，更新一次 actor
    batch_size: int = 32                     # 每次更新采样的 batch 大小
    replay_size: int = 100_000               # 经验回放容量
    total_steps: int = 50_000                # 训练总迭代步数（按 learner 循环计数）
    start_steps: int = 1_000                 # 前多少样本用纯探索策略（不依赖 actor）
    eval_interval: int = 5_000               # 评估间隔（learner 步数）
    eval_episodes: int = 3                   # 每次评估跑多少 episode
    save_models: bool = True                 # 是否保存模型
    save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")  # 模型保存目录
    load_run_id: Optional[str] = None        # 若指定，则从该 run_id 目录加载模型
    save_best: bool = True                   # eval 表现更好时是否保存 best_*.pt
    early_stop_patience: int = 0             # 若 >0，eval reward 多次未提升则提前停止
    min_steps_for_early_stop: int = 0        # 早停生效的最小训练步数
    early_stop_fC_threshold: float = 0.0     # 仅当 eval mean_fC 高于该阈值才允许早停
    lr_decay_fC_threshold: float = 0.7       # eval 合作率达到阈值时触发 lr 衰减
    lr_decay_multiplier: float = 0.5         # lr 衰减乘子
    batch_envs: int = 8                      # 单进程向量化环境的并行数量
    updates_per_step: int | None = None      # 每个 learner step 做多少次更新，None 时默认等于 batch_envs


class ReplayBuffer:
    """基于 numpy 的环形缓冲区，支持 O(1) 采样。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def add(self, s, a, r, s2, done):
        s_arr = np.asarray(s, dtype=np.float32)
        a_arr = np.asarray(a, dtype=np.float32)
        s2_arr = np.asarray(s2, dtype=np.float32)
        r_val = float(r)
        d_val = float(done)
        if self.states is None:
            self.states = np.empty((self.capacity, *s_arr.shape), dtype=np.float32)
            self.actions = np.empty((self.capacity, *a_arr.shape), dtype=np.float32)
            self.rewards = np.empty((self.capacity,), dtype=np.float32)
            self.next_states = np.empty((self.capacity, *s2_arr.shape), dtype=np.float32)
            self.dones = np.empty((self.capacity,), dtype=np.float32)
        idx = self.ptr
        self.states[idx] = s_arr
        self.actions[idx] = a_arr
        self.rewards[idx] = r_val
        self.next_states[idx] = s2_arr
        self.dones[idx] = d_val
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def __len__(self):
        return self.size

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


def select_action(actor: ActorNet, state: np.ndarray, device: str, expl_noise: float, training: bool) -> np.ndarray:
    """
    state: (C,L,L) 或 (B,C,L,L)
    返回 pi_field: (L,L,5) 或 (B,L,L,5)
    """
    actor.eval()
    with torch.no_grad():
        s_np = np.asarray(state, dtype=np.float32)
        if s_np.ndim == 3:
            s_np = s_np[None, ...]
        s_tensor = torch.from_numpy(s_np).float().to(device)
        feat = actor.body(s_tensor)
        logits = actor.policy_head(feat)
        if training and expl_noise > 0.0:
            noise = torch.randn_like(logits) * expl_noise
            logits = logits + noise
        pi = torch.softmax(logits, dim=1)
        pi_np = pi.cpu().numpy()  # (B,5,L,L)
        pi_np = np.transpose(pi_np, (0, 2, 3, 1))  # (B,L,L,5)
    actor.train()
    return pi_np[0] if pi_np.shape[0] == 1 else pi_np


def train_td3(
    L: int = 32,
    r: float = 1.4,
    episode_length: int = 500,
    cfg: TD3Config | None = None,
    initial_R: float | None = None,
):
    if cfg is None:
        cfg = TD3Config()
    device = cfg.device

    env_kwargs = dict(
        L=L,
        r=r,
        episode_length=episode_length,
        use_cumulative_planner_reward=False,
    )
    if initial_R is not None:
        env_kwargs["initial_R"] = initial_R
    # 批量环境：单进程内并行 batch_envs 个棋盘
    env = BatchedPublicGoodsEnv(batch_size=cfg.batch_envs, **env_kwargs)
    # 评估环境按需新建，确保与训练环境参数一致
    eval_env = PublicGoodsEnv(**env_kwargs)

    actor = ActorNet(in_channels=STATE_CHANNELS).to(device)
    actor_target = ActorNet(in_channels=STATE_CHANNELS).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic1 = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic2 = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic1_target = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic2_target = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    actor_sched = MultiplicativeLR(actor_opt, lr_lambda=lambda _: cfg.lr_decay_multiplier)
    critic_sched = MultiplicativeLR(critic_opt, lr_lambda=lambda _: cfg.lr_decay_multiplier)
    init_actor_lr = actor_opt.param_groups[0]["lr"]
    init_critic_lr = critic_opt.param_groups[0]["lr"]

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.save_dir, run_id)
    if cfg.save_models:
        os.makedirs(run_dir, exist_ok=True)

    if cfg.load_run_id is not None:
        load_dir = os.path.join(cfg.save_dir, cfg.load_run_id)
        try:
            actor.load_state_dict(torch.load(os.path.join(load_dir, "actor.pt"), map_location=device))
            actor_target.load_state_dict(torch.load(os.path.join(load_dir, "actor_target.pt"), map_location=device))
            critic1.load_state_dict(torch.load(os.path.join(load_dir, "critic1.pt"), map_location=device))
            critic2.load_state_dict(torch.load(os.path.join(load_dir, "critic2.pt"), map_location=device))
            critic1_target.load_state_dict(torch.load(os.path.join(load_dir, "critic1_target.pt"), map_location=device))
            critic2_target.load_state_dict(torch.load(os.path.join(load_dir, "critic2_target.pt"), map_location=device))
            print(f"[TD3] Loaded models from run_id={cfg.load_run_id}")
        except FileNotFoundError:
            print(f"[TD3] Warning: checkpoint for run_id={cfg.load_run_id} not found, start from scratch.")

    replay = ReplayBuffer(cfg.replay_size)

    step_rewards = []
    step_fCs = []
    eval_rewards = []
    eval_fCs = []
    actor_loss_hist = []
    critic_loss_hist = []
    best_eval_reward = -float("inf")
    since_best = 0
    stop_training = False
    lr_lowered = False

    log_file = None
    log_writer = None
    if cfg.save_models:
        log_path = os.path.join(run_dir, "training_log.csv")
        log_file = open(log_path, "w", newline="")
        log_writer = csv.writer(log_file)
        tb_writer = SummaryWriter(run_dir)
        log_writer.writerow([
            "type", "step",
            "episode_reward", "mean_episode_fC",
            "actor_loss", "critic_loss",
            "eval_reward", "eval_mean_fC",
        ])
    else:
        os.makedirs(run_dir, exist_ok=True)
        tb_writer = None

    sample_count = 0        # 已采集的样本总数（按 batch_envs 计）
    update_step = 0         # 已进行的 critic/actor 更新计数
    state = env.reset()  # (B,4,L,L)
    ep_reward_sum = 0.0
    ep_fC_sum = 0.0
    ep_len = 0

    for step in range(1, cfg.total_steps + 1):
        if stop_training:
            break

        # 1) 环境采样：并行推进 batch_envs 个棋盘一步
        if sample_count < cfg.start_steps:
            pi_field = np.random.dirichlet(np.ones(5), size=(cfg.batch_envs, L, L)).astype(np.float32)
        else:
            pi_field = select_action(actor, state, device, cfg.expl_noise, training=True)  # (B,L,L,5)
        next_state, reward_batch, done, info = env.step(pi_field)

        # 2) 写入 replay（逐 env）
        # 存入 replay：训练目标视角，不考虑时间截断，done 统一按 False 处理
        for b in range(cfg.batch_envs):
            replay.add(state[b], pi_field[b], reward_batch[b], next_state[b], False)
        sample_count += cfg.batch_envs

        # 3) 累积当前 episode 的平均 reward / f_C（按 batch 均值）
        ep_reward_sum += float(np.mean(reward_batch))
        ep_fC_sum += float(info.get("f_C", 0.0))
        ep_len += 1
        state = next_state if not done else env.reset()

        if step % 5000 == 0:
            progress = 100.0 * step / float(cfg.total_steps)
            print(f"[TD3] step={step}/{cfg.total_steps} ({progress:.2f}%), samples={sample_count}, replay_size={len(replay)}")

        if done:
            mean_fC = ep_fC_sum / max(1, ep_len)
            step_rewards.append(ep_reward_sum)
            step_fCs.append(mean_fC)
            if log_writer is not None:
                log_writer.writerow([
                    "train_episode",
                    step,
                    ep_reward_sum,
                    mean_fC,
                    "",
                    "",
                    "",
                    "",
                ])
            if tb_writer is not None:
                tb_writer.add_scalar("train/episode_reward", ep_reward_sum, len(step_rewards))
                tb_writer.add_scalar("train/mean_fC", mean_fC, len(step_rewards))
            ep_reward_sum = 0.0
            ep_fC_sum = 0.0
            ep_len = 0

        # 4) Replay 足够且过了 warmup 后才做一次更新
        if len(replay) >= cfg.batch_size and sample_count >= cfg.start_steps:
            # 根据配置决定每个 learner step 做多少次更新，默认与 batch_envs 相同
            updates_this_step = cfg.updates_per_step or cfg.batch_envs
            for _ in range(updates_this_step):
                s_np, a_np, r_np, s2_np, d_np = replay.sample(cfg.batch_size)

                s = torch.from_numpy(s_np).float().to(device)
                a = torch.from_numpy(a_np).float().to(device)
                r_batch = torch.from_numpy(r_np).float().to(device)
                s2 = torch.from_numpy(s2_np).float().to(device)
                d = torch.from_numpy(d_np).float().to(device)

                a_t = a.permute(0, 3, 1, 2)  # (B,5,L,L)

                with torch.no_grad():
                    feat2 = actor_target.body(s2)
                    logits2 = actor_target.policy_head(feat2)
                    if cfg.policy_noise > 0.0:
                        noise = torch.randn_like(logits2) * cfg.policy_noise
                        noise = torch.clamp(noise, -cfg.noise_clip, cfg.noise_clip)
                        logits2 = logits2 + noise
                    a2 = torch.softmax(logits2, dim=1)

                    q1_target = critic1_target(s2, a2)
                    q2_target = critic2_target(s2, a2)
                    q_target = torch.min(q1_target, q2_target)
                    target = r_batch + cfg.gamma * (1.0 - d) * q_target

                q1 = critic1(s, a_t)
                q2 = critic2(s, a_t)
                critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()

                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 40.0)
                critic_opt.step()

                critic_loss_hist.append(float(critic_loss.item()))

                update_step += 1
                if update_step % cfg.policy_delay == 0:
                    pi = actor(s)
                    actor_loss = -critic1(s, pi).mean()

                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 40.0)
                    actor_opt.step()

                    actor_loss_hist.append(float(actor_loss.item()))

                    soft_update(actor_target, actor, cfg.tau)
                    soft_update(critic1_target, critic1, cfg.tau)
                    soft_update(critic2_target, critic2, cfg.tau)

        # 5) 定期评估（单环境复用 eval_env，关闭探索噪声）
        if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
            eval_env = PublicGoodsEnv(**env_kwargs)
            actor.eval()
            n_eval_ep = max(1, cfg.eval_episodes)
            total_er, total_ef = 0.0, 0.0
            for _ in range(n_eval_ep):
                s_eval = eval_env.reset()
                ep_r = 0.0
                ep_fC_sum_eval = 0.0
                ep_len_eval = 0
                done_eval = False
                while not done_eval and ep_len_eval < episode_length:
                    pi_eval = select_action(actor, s_eval, device, expl_noise=0.0, training=False)
                    s_eval, r_eval, done_eval, info_eval = eval_env.step(pi_eval)
                    ep_r += r_eval
                    ep_fC_sum_eval += float(info_eval.get("f_C", 0.0))
                    ep_len_eval += 1
                total_er += ep_r
                total_ef += ep_fC_sum_eval / max(1, ep_len_eval)
            actor.train()

            mean_er = total_er / n_eval_ep
            mean_ef = total_ef / n_eval_ep
            eval_rewards.append(mean_er)
            eval_fCs.append(mean_ef)
            print(f"[Eval] step={step}, eval_reward={mean_er:.4f}, eval_mean_fC={mean_ef:.4f}")
            if (not lr_lowered) and (mean_ef > cfg.lr_decay_fC_threshold):
                actor_sched.step()
                critic_sched.step()
                lr_lowered = True
                print(
                    f"[LR] eval_mean_fC={mean_ef:.4f} > {cfg.lr_decay_fC_threshold}, 降低学习率，"
                    f"actor_lr={actor_opt.param_groups[0]['lr']:.6g}, "
                    f"critic_lr={critic_opt.param_groups[0]['lr']:.6g}"
                )
            if mean_er > best_eval_reward + 1e-8:
                best_eval_reward = mean_er
                since_best = 0
                if cfg.save_models and cfg.save_best:
                    torch.save(actor.state_dict(), os.path.join(run_dir, "best_actor.pt"))
                    torch.save(actor_target.state_dict(), os.path.join(run_dir, "best_actor_target.pt"))
                    torch.save(critic1.state_dict(), os.path.join(run_dir, "best_critic1.pt"))
                    torch.save(critic2.state_dict(), os.path.join(run_dir, "best_critic2.pt"))
                    torch.save(critic1_target.state_dict(), os.path.join(run_dir, "best_critic1_target.pt"))
                    torch.save(critic2_target.state_dict(), os.path.join(run_dir, "best_critic2_target.pt"))
                    print(f"[Eval] Saved new best models at step={step} (reward={mean_er:.4f})")
            else:
                since_best += 1
                if (
                    cfg.early_stop_patience > 0
                    and step >= cfg.min_steps_for_early_stop
                    and since_best >= cfg.early_stop_patience
                    and mean_ef >= cfg.early_stop_fC_threshold
                ):
                    print(
                        f"[Eval] Early stopping triggered at step={step} "
                        f"(no improvement for {since_best} evals, min_steps={cfg.min_steps_for_early_stop}, "
                        f"mean_fC={mean_ef:.4f} >= early_stop_fC_threshold={cfg.early_stop_fC_threshold})"
                    )
                    stop_training = True
            if log_writer is not None:
                log_writer.writerow([
                    "eval",
                    step,
                    "",
                    "",
                    "",
                    "",
                    mean_er,
                    mean_ef,
                ])
            if tb_writer is not None:
                tb_writer.add_scalar("eval/reward", mean_er, step)
                tb_writer.add_scalar("eval/mean_fC", mean_ef, step)

    if cfg.save_models:
        torch.save(actor.state_dict(), os.path.join(run_dir, "actor.pt"))
        torch.save(actor_target.state_dict(), os.path.join(run_dir, "actor_target.pt"))
        torch.save(critic1.state_dict(), os.path.join(run_dir, "critic1.pt"))
        torch.save(critic2.state_dict(), os.path.join(run_dir, "critic2.pt"))
        torch.save(critic1_target.state_dict(), os.path.join(run_dir, "critic1_target.pt"))
        torch.save(critic2_target.state_dict(), os.path.join(run_dir, "critic2_target.pt"))
        print(f"[TD3] Saved models to directory={run_dir}")

    title_suffix = f"L={L}, r={r}, steps={cfg.total_steps}, bs={cfg.batch_size}"

    if plt is not None and len(step_rewards) > 0:
        xs = list(range(len(step_rewards)))
        plt.figure()
        plt.plot(xs, step_rewards, label="episode reward")
        plt.xlabel("episode index")
        plt.ylabel("reward")
        plt.title(f"TD3: episode rewards ({title_suffix})")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(run_dir, "td3_episode_rewards.png") if cfg.save_models else "td3_episode_rewards.png"
        plt.savefig(out_path)
        plt.close()

        xs_f = list(range(len(step_fCs)))
        plt.figure()
        plt.plot(xs_f, step_fCs, label="episode mean f_C")
        plt.xlabel("episode index")
        plt.ylabel("cooperation rate")
        plt.title(f"TD3: episode mean f_C ({title_suffix})")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(run_dir, "td3_episode_fC.png") if cfg.save_models else "td3_episode_fC.png"
        plt.savefig(out_path)
        plt.close()

    if plt is not None and len(eval_rewards) > 0:
        xs = [cfg.eval_interval * (i + 1) for i in range(len(eval_rewards))]
        plt.figure()
        plt.plot(xs, eval_rewards, label="eval reward")
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.title(f"TD3: eval rewards ({title_suffix})")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(run_dir, "td3_eval_rewards.png") if cfg.save_models else "td3_eval_rewards.png"
        plt.savefig(out_path)
        plt.close()

        plt.figure()
        plt.plot(xs, eval_fCs, label="eval mean f_C")
        plt.xlabel("step")
        plt.ylabel("cooperation rate")
        plt.title(f"TD3: eval mean f_C ({title_suffix})")
        plt.grid(True)
        plt.tight_layout()
        out_path = os.path.join(run_dir, "td3_eval_fC.png") if cfg.save_models else "td3_eval_fC.png"
        plt.savefig(out_path)
        plt.close()

    if log_file is not None:
        log_file.close()
    if tb_writer is not None:
        tb_writer.close()


def evaluate_trained_actor(
    actor_path: str,
    L: int,
    r: float,
    episode_length: int = 500,
    eval_episodes: int = 5,
    device: str = "cpu",
    initial_R: float | None = None,
    R_decay: float = 0.10,
    coop_cost: float = 5.0,
    use_cumulative_planner_reward: bool = False,
):
    actor = ActorNet(in_channels=STATE_CHANNELS).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    env_kwargs = dict(
        L=L,
        r=r,
        episode_length=episode_length,
        use_cumulative_planner_reward=use_cumulative_planner_reward,
        R_decay=R_decay,
        coop_cost=coop_cost,
    )
    if initial_R is not None:
        env_kwargs["initial_R"] = initial_R
    env = PublicGoodsEnv(**env_kwargs)
    rewards = []
    fcs = []
    for _ in range(eval_episodes):
        state = env.reset()
        ep_reward = 0.0
        ep_fC_sum = 0.0
        ep_len = 0
        done = False
        while not done and ep_len < episode_length:
            pi_field = select_action(actor, state, device, expl_noise=0.0, training=False)
            state, reward, done, info = env.step(pi_field)
            ep_reward += reward
            ep_fC_sum += float(info.get("f_C", 0.0))
            ep_len += 1
        rewards.append(ep_reward)
        fcs.append(ep_fC_sum / max(1, ep_len))
    mean_reward = float(np.mean(rewards))
    mean_fC = float(np.mean(fcs))
    print(f"[Eval] actor={actor_path}, L={L}, r={r}, mean_reward={mean_reward:.4f}, mean_fC={mean_fC:.4f}")
    return mean_reward, mean_fC


if __name__ == "__main__":
    #训练的参数配置
    cfg1 = TD3Config(
        device="cuda:0",                    # 使用的 GPU/CPU
        gamma=0.99,                         # 折扣因子
        actor_lr=1e-4,                      # Actor 学习率
        critic_lr=1e-4,                     # Critic 学习率
        tau=0.005,                          # target 网络软更新系数
        policy_noise=0.2,                   # target policy smoothing 噪声0.03
        noise_clip=0.5,                     # target 噪声截断范围0.05
        expl_noise=0.1,                     # 行为策略探索噪声=0.005
        policy_delay=2,                     # 每 2 次 critic 更新做 1 次 actor 更新
        batch_size=512,                    # 训练时的 batch 大小
        replay_size=100_000,                # replay buffer 容量
        total_steps=1_000_000,              # learner 循环总步数
        start_steps=15_000,                 # warmup 样本阈值
        eval_interval=5_000,                # 评估间隔
        eval_episodes=5,                    # 每次评估的 episode 数
        save_models=True,                   # 是否保存模型
        save_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),  # 模型保存目录
        load_run_id=None,                   # 若从已有 run 续训，则填 run_id
        save_best=True,                     # eval 提升时保存 best_*.pt
        early_stop_patience=10,             # eval 多次未提升时提前停止
        min_steps_for_early_stop=600_000,   # 早停生效的最小步数
        early_stop_fC_threshold=0.7,        # eval 合作率高于该阈值才允许早停
        lr_decay_fC_threshold=0.7,          # eval 合作率达阈值时触发 lr 衰减
        lr_decay_multiplier=0.25,           # lr 衰减乘子
        batch_envs=16,                      # 单进程环境并行数
        updates_per_step=1,              # 每个 learner step 做多少次更新，None 时默认等于 batch_envs

    )

    # 如需仅评估已有模型，配置以下开关和参数
    EVAL_ONLY = False
    EVAL_RUN_ID = ""  # 如 "20251119_230050第一版T3D较好效果"
    EVAL_L_LIST = [32]
    EVAL_R_LIST = [1.4]
    EVAL_EPISODE_LENGTH_LIST = [500]
    EVAL_EPISODES = 5
    EVAL_DEVICE = "cpu"
    EVAL_INITIAL_R = None      # 若为 None 则用环境默认值；否则覆盖
    EVAL_R_DECAY = 0.10
    EVAL_COOP_COST = 5.0
    EVAL_USE_CUM_REWARD = False

    if EVAL_ONLY:
        if not EVAL_RUN_ID:
            raise ValueError("EVAL_ONLY=True 时必须设置 EVAL_RUN_ID")
        actor_path = os.path.join(cfg1.save_dir, EVAL_RUN_ID, "actor.pt")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_csv = os.path.join(cfg1.save_dir, f"eval_results_{EVAL_RUN_ID}_{timestamp}.csv")

        results = []
        for L_eval in EVAL_L_LIST:
            for r_eval in EVAL_R_LIST:
                for ep_len in EVAL_EPISODE_LENGTH_LIST:
                    mean_reward, mean_fC = evaluate_trained_actor(
                        actor_path=actor_path,
                        L=L_eval,
                        r=r_eval,
                        episode_length=ep_len,
                        eval_episodes=EVAL_EPISODES,
                        device=EVAL_DEVICE,
                        initial_R=EVAL_INITIAL_R,
                        R_decay=EVAL_R_DECAY,
                        coop_cost=EVAL_COOP_COST,
                        use_cumulative_planner_reward=EVAL_USE_CUM_REWARD,
                    )
                    results.append((L_eval, r_eval, ep_len, EVAL_EPISODES, mean_reward, mean_fC))

        with open(eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["L", "r", "episode_length", "eval_episodes", "mean_reward", "mean_fC"])
            for row in results:
                writer.writerow(row)
        print(f"[Eval] 保存所有组合结果到 {eval_csv}")
    else:
        train_td3(
            L=32,
            r=4,
            episode_length=150,
            cfg=cfg1,
        )
