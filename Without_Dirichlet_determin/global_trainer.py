"""
TD3 trainer for the deterministic planner in the spatial public-goods game.

入口：在本目录下运行

    python -m global_trainer
或
    python global_trainer.py

本实现是单环境、单进程版本的 TD3：
- ActorNet: 确定性输出整张棋盘的分配比例场 pi_field；
- CriticNet: 双 Q 网络 Q1, Q2；
- 经验回放 + target 网络 + policy 延迟更新。
"""

from __future__ import annotations

import random
import multiprocessing as mp
from multiprocessing.queues import Queue as MPQueue
from multiprocessing.synchronize import Event as MPEvent
from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import os
import datetime
import csv
import csv

# 限制单进程内部的线程数，避免多进程采样时把所有 CPU 占满
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.tensorboard import SummaryWriter
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from env import PublicGoodsEnv
from planner_net import ActorNet, CriticNet


@dataclass
class TD3Config:
    """
    TD3 超参数配置。

    可以在 __main__ 里创建一个 TD3Config 实例并传给 train_td3。
    """

    device: str = "cpu"          # 训练设备："cpu" 或 "cuda:0" 等
    gamma: float = 0.99          # 折扣因子
    actor_lr: float = 1e-4       # Actor 学习率
    critic_lr: float = 1e-4      # Critic 学习率
    tau: float = 0.005           # target 网络软更新系数
    policy_noise: float = 0.1    # target smoothing 噪声强度（加在 target actor logits 上）
    noise_clip: float = 0.2      # target smoothing 噪声截断范围
    expl_noise: float = 0      # 行为策略探索噪声（加在行为 logits 上）；这个值应该很小0.01
    policy_delay: int = 2        # 每多少次 critic 更新，更新一次 actor
    batch_size: int = 32         # 每次更新时从 replay buffer 采样的 batch 大小
    replay_size: int = 100_000   # replay buffer 容量
    total_steps: int = 50_000    # 总环境交互步数（time steps）
    start_steps: int = 1_000     # 前若干步使用纯探索策略（不依赖 actor）
    eval_interval: int = 5_000   # 每多少步做一次评估
    eval_episodes: int = 3       # 每次评估跑多少个 episode
    save_models: bool = True     # 是否在训练过程中保存模型
    save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")  # 模型保存的根目录
    load_run_id: str | None = None  # 若不为 None，则尝试从该 run_id 加载已有模型
    save_best: bool = True       # eval 表现更好时是否保存 best_*.pt
    early_stop_patience: int = 0 # 若 >0，则 eval reward 连续若干次未提升时提前停止
    min_steps_for_early_stop: int = 0  # 早停生效的最小 step，默认 0 表示从一开始就生效
    early_stop_fC_threshold: float = 0.0  # 仅当 eval mean_fC 高于该阈值时才允许早停
    lr_decay_fC_threshold: float = 0.7  # eval 合作率达到该阈值时触发 lr 衰减
    lr_decay_multiplier: float = 0.5    # lr 衰减乘子（乘在当前 lr 上）
    rollout_workers: int | None = None  # 并行采样进程数；None 表示自动取 cpu_count 范围内的值
    samples_per_step: int | None = None  # 每个训练 step 从 data_queue 最多取多少条样本；None 默认等于 rollout_workers 或 4


class ReplayBuffer:
    """
    简单的经验回放缓冲区：
    存储 (state, action, reward, next_state, done) 五元组。

    为了避免 deque + random.sample 在大容量下的 O(N) 开销，
    这里使用基于 numpy 的环形缓冲区，实现 O(1) 随机索引与向量化采样。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        # 延迟初始化存储数组，首次 add 时按样本形状分配
        self.states: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self.rewards: np.ndarray | None = None
        self.next_states: np.ndarray | None = None
        self.dones: np.ndarray | None = None

    def add(self, s, a, r, s2, done):
        """追加一个 transition 到缓冲区。"""
        s_arr = np.asarray(s, dtype=np.float32)
        a_arr = np.asarray(a, dtype=np.float32)
        s2_arr = np.asarray(s2, dtype=np.float32)
        r_val = float(r)
        d_val = float(done)

        if self.states is None:
            # 首次调用时根据样本形状分配环形缓冲区
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
        """随机采样一个 batch，并按 numpy 数组打包返回。"""
        assert self.size > 0 and self.states is not None
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )


@dataclass
class Transition:
    """跨进程传递的单步样本。"""
    s: np.ndarray
    a: np.ndarray
    r: float
    s2: np.ndarray
    done: bool


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)


STATE_CHANNELS = 4  # stra_now, stra_prev, P_center_norm, R_norm


def select_action(actor: ActorNet, state: np.ndarray, device: str, expl_noise: float, training: bool) -> np.ndarray:
    """
    用 actor 产生一个确定性动作，并在训练时在 logits 上加一点高斯噪声进行探索。

    参数:
        actor: 训练中的 ActorNet
        state: numpy 数组 (STATE_CHANNELS, L, L)，当前棋盘状态
        device: 运行设备标识
        expl_noise: 探索噪声标准差（加在 logits 上）
        training: 若为 True，则加噪声；评估时可设为 False

    返回:
        pi_field: numpy 数组 (L, L, 5)，每格点的分配比例
    """
    actor.eval()
    with torch.no_grad():
        s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1,C,L,L)
        feat = actor.body(s_tensor)  # 直接访问 body + policy_head，便于在 logits 上加噪声
        logits = actor.policy_head(feat)  # (1,5,L,L)
        if training and expl_noise > 0.0:
            noise = torch.randn_like(logits) * expl_noise
            logits = logits + noise
        pi = torch.softmax(logits, dim=1)  # (1,5,L,L)
        pi_np = pi.cpu().numpy()[0]  # (5,L,L)
        pi_np = np.transpose(pi_np, (1, 2, 0))  # -> (L,L,5)，env.step 需要这种格式
    actor.train()
    return pi_np


def rollout_worker(
    worker_id: int,
    env_kwargs: dict,
    expl_noise: float,
    data_queue: MPQueue,
    param_queue: MPQueue,
    metric_queue: MPQueue,
    stop_event: MPEvent,
):
    """
    多进程采样 worker：在 CPU 上跑一个 env，使用收到的最新 actor 参数生成样本。
    """
    env = PublicGoodsEnv(**env_kwargs)
    actor = ActorNet(in_channels=STATE_CHANNELS).to("cpu")
    actor.eval()

    # 初次拉取参数（阻塞等待）
    state_dict = param_queue.get()
    actor.load_state_dict(state_dict)

    state = env.reset()
    ep_reward = 0.0
    ep_fC_sum = 0.0
    ep_len = 0
    while not stop_event.is_set():
        # 如有新参数，更新本地 actor
        try:
            while True:
                state_dict = param_queue.get_nowait()
                actor.load_state_dict(state_dict)
        except Exception:
            pass

        pi_field = select_action(actor, state, device="cpu", expl_noise=expl_noise, training=True)
        next_state, reward, done, info = env.step(pi_field)

        tr = Transition(
            s=state.astype(np.float32),
            a=pi_field.astype(np.float32),
            r=float(reward),
            s2=next_state.astype(np.float32),
            done=bool(done),
        )
        data_queue.put(tr)

        ep_reward += float(reward)
        ep_fC_sum += float(info.get("f_C", 0.0))
        ep_len += 1

        if done:
            mean_fC = ep_fC_sum / max(1, ep_len)
            metric_queue.put((worker_id, ep_reward, mean_fC))
            state = env.reset()
            ep_reward = 0.0
            ep_fC_sum = 0.0
            ep_len = 0
        else:
            state = next_state

    # 退出前清空一次参数队列，避免主进程阻塞（可选）
    try:
        while True:
            param_queue.get_nowait()
    except Exception:
        pass


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

    # 这里开启累计式 planner 奖励，直接使用每一步平均净收益作为 reward
    env_kwargs = dict(
        L=L,
        r=r,
        episode_length=episode_length,
        # TD3 用即时奖励，关闭累计式奖励避免非平稳 target
        use_cumulative_planner_reward=False,
    )
    # 如指定 initial_R，则覆盖环境默认初始资源
    if initial_R is not None:
        env_kwargs["initial_R"] = initial_R

    # 主网络：Actor 和两个 Critic
    actor = ActorNet(in_channels=STATE_CHANNELS).to(device)
    actor_target = ActorNet(in_channels=STATE_CHANNELS).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic1 = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic2 = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic1_target = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic2_target = CriticNet(state_channels=STATE_CHANNELS).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    # 优化器：一个 Actor，一个共享 Critic 优化器（同时更新 Q1/Q2）
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=cfg.critic_lr)
    # 学习率调度器：当 eval 合作率达到阈值时手动降低 lr（乘以 0.5）
    actor_sched = MultiplicativeLR(actor_opt, lr_lambda=lambda _: cfg.lr_decay_multiplier)
    critic_sched = MultiplicativeLR(critic_opt, lr_lambda=lambda _: cfg.lr_decay_multiplier)
    # 记录初始学习率，便于阈值恢复
    init_actor_lr = actor_opt.param_groups[0]["lr"]
    init_critic_lr = critic_opt.param_groups[0]["lr"]

    # 运行 ID（用于保存模型与图像），时间戳形式
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.save_dir, run_id)
    if cfg.save_models:
        os.makedirs(run_dir, exist_ok=True)

    # 如指定 load_run_id，则尝试从已有 checkpoint 加载模型参数
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

    # 多进程采样队列与 worker 管理
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start_method 可能已在主进程设过，忽略
        pass

    num_workers = cfg.rollout_workers or max(1, min(8, mp.cpu_count()))
    data_queue: MPQueue = mp.Queue(maxsize=10_000)
    param_queue: MPQueue = mp.Queue()
    metric_queue: MPQueue = mp.Queue(maxsize=10_000)
    stop_event: MPEvent = mp.Event()

    def _actor_state_dict_cpu():
        return {k: v.detach().cpu() for k, v in actor.state_dict().items()}

    # 初次广播 actor 参数给所有 worker
    init_sd = _actor_state_dict_cpu()
    for _ in range(num_workers):
        param_queue.put(init_sd)

    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=rollout_worker,
            args=(wid, env_kwargs, cfg.expl_noise, data_queue, param_queue, metric_queue, stop_event),
        )
        p.daemon = True
        p.start()
        workers.append(p)
    print(f"[TD3] Started {num_workers} rollout workers.")

    replay = ReplayBuffer(cfg.replay_size)

    # 记录指标
    step_rewards = []      # 每个 episode 的总奖励
    step_fCs = []          # 每个 episode 的平均合作率
    eval_rewards = []      # eval 阶段的平均奖励
    eval_fCs = []          # eval 阶段的平均合作率
    actor_loss_hist = []   # 训练过程中 Actor loss 序列
    critic_loss_hist = []  # 训练过程中 Critic loss 序列
    best_eval_reward = -float("inf")
    since_best = 0
    stop_training = False
    lr_lowered = False  # 只在 eval 合作率高于阈值时降低一次 lr

    # 打开训练日志（CSV）
    log_file = None
    log_writer = None
    if cfg.save_models:
        os.makedirs(run_dir, exist_ok=True)
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

    # 采样计数（用于 warmup）
    sample_count = 0
    update_step = 0
    mean_er, mean_ef = float("nan"), float("nan")  # eval 占位，避免未定义
    param_broadcast_interval = 1_000  # 主网络参数下发给 workers 的间隔步数

    for step in range(1, cfg.total_steps + 1):
        if stop_training:
            break

        # 1) 从 data_queue 拉取若干条样本填充 replay
        #    单步最多拉取与并行 worker 数量同量级的样本，避免队列被迅速塞满导致 worker 阻塞
        max_fetch = cfg.samples_per_step or (cfg.rollout_workers or 4)
        fetched = 0
        while fetched < max_fetch:
            try:
                tr = data_queue.get(timeout=0.01)
            except Exception:
                break
            replay.add(tr.s, tr.a, tr.r, tr.s2, tr.done)
            sample_count += 1
            fetched += 1

        if step % 5000 == 0:
            progress = 100.0 * step / float(cfg.total_steps)
            print(f"[TD3] step={step}/{cfg.total_steps} ({progress:.2f}%), samples={sample_count}, replay_size={len(replay)}")

        # 1.5) 拉取 worker 上报的 episode 统计，写入日志/TensorBoard
        metrics_fetched = 0
        while metrics_fetched < 10:
            try:
                wid, ep_r, ep_fC = metric_queue.get_nowait()
            except Exception:
                break
            metrics_fetched += 1
            step_rewards.append(ep_r)
            step_fCs.append(ep_fC)
            if log_writer is not None:
                log_writer.writerow([
                    "train_episode",
                    step,
                    ep_r,
                    ep_fC,
                    "",
                    "",
                    "",
                    "",
                ])
            if tb_writer is not None:
                tb_writer.add_scalar("train/episode_reward", ep_r, len(step_rewards))
                tb_writer.add_scalar("train/mean_fC", ep_fC, len(step_rewards))

        # 2) 更新网络（当 replay 中样本数量足够时）
        if len(replay) >= cfg.batch_size and sample_count >= cfg.start_steps:
            # 3.1 采样 batch
            s_np, a_np, r_np, s2_np, d_np = replay.sample(cfg.batch_size)

            s = torch.from_numpy(s_np).float().to(device)       # (B,STATE_CHANNELS,L,L)
            a = torch.from_numpy(a_np).float().to(device)       # (B,L,L,5)
            r_batch = torch.from_numpy(r_np).float().to(device)       # (B,)
            s2 = torch.from_numpy(s2_np).float().to(device)     # (B,STATE_CHANNELS,L,L)
            d = torch.from_numpy(d_np).float().to(device)       # (B,)

            # 转动作为 (B,5,L,L)
            a_t = a.permute(0, 3, 1, 2)  # (B,5,L,L)

            # 3.2 计算 target Q（TD3 的 target policy smoothing）
            with torch.no_grad():
                # target actor 输出 next action，加入小噪声做 target smoothing
                feat2 = actor_target.body(s2)
                logits2 = actor_target.policy_head(feat2)
                if cfg.policy_noise > 0.0:
                    noise = torch.randn_like(logits2) * cfg.policy_noise
                    noise = torch.clamp(noise, -cfg.noise_clip, cfg.noise_clip)
                    logits2 = logits2 + noise
                a2 = torch.softmax(logits2, dim=1)  # (B,5,L,L)

                q1_target = critic1_target(s2, a2)
                q2_target = critic2_target(s2, a2)
                q_target = torch.min(q1_target, q2_target)
                target = r_batch + cfg.gamma * (1.0 - d) * q_target

            # 3.3 更新两个 critic
            q1 = critic1(s, a_t)
            q2 = critic2(s, a_t)
            critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 40.0)
            critic_opt.step()

            critic_loss_hist.append(float(critic_loss.item()))

            update_step += 1
            # 3.4 延迟更新 actor 和 target 网络（policy delay）
            if update_step % cfg.policy_delay == 0:
                # actor_loss = - E_s [ Q1(s, pi(s)) ]
                pi = actor(s)  # (B,5,L,L)
                actor_loss = -critic1(s, pi).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 40.0)
                actor_opt.step()

                actor_loss_hist.append(float(actor_loss.item()))

                soft_update(actor_target, actor, cfg.tau)
                soft_update(critic1_target, critic1, cfg.tau)
                soft_update(critic2_target, critic2, cfg.tau)

        # 3) 定期把最新 actor 参数广播给 workers
        if step % param_broadcast_interval == 0:
            cpu_sd = {k: v.detach().cpu() for k, v in actor.state_dict().items()}
            for _ in range(num_workers):
                param_queue.put(cpu_sd)

        # 4) 定期做 eval（关掉探索噪声，评估当前策略表现）
        if cfg.eval_interval > 0 and step % cfg.eval_interval == 0:
            # 使用一个新的 env 做评估，避免干扰训练 env / replay
            eval_env = PublicGoodsEnv(
                L=L, r=r, episode_length=episode_length, use_cumulative_planner_reward=False
            )
            actor.eval()
            n_eval_ep = max(1, cfg.eval_episodes)
            total_er, total_ef = 0.0, 0.0
            for _ in range(n_eval_ep):
                s_eval = eval_env.reset()
                ep_r = 0.0
                ep_fC_sum = 0.0
                ep_len = 0
                done_eval = False
                while not done_eval and ep_len < episode_length:
                    # 评估时不加探索噪声（expl_noise=0, training=False）
                    pi_eval = select_action(actor, s_eval, device, expl_noise=0.0, training=False)
                    s_eval, r_eval, done_eval, info_eval = eval_env.step(pi_eval)
                    ep_r += r_eval
                    ep_fC_sum += float(info_eval.get("f_C", 0.0))
                    ep_len += 1
                total_er += ep_r
                total_ef += ep_fC_sum / max(1, ep_len)
            actor.train()

            mean_er = total_er / n_eval_ep
            mean_ef = total_ef / n_eval_ep
            eval_rewards.append(mean_er)
            eval_fCs.append(mean_ef)
            print(
                f"[Eval] step={step}, eval_reward={mean_er:.4f}, eval_mean_fC={mean_ef:.4f}"
            )
            # 若合作率高于阈值且尚未降低过 lr，则将 actor/critic lr 乘以衰减系数
            if (not lr_lowered) and (mean_ef > cfg.lr_decay_fC_threshold):
                actor_sched.step()
                critic_sched.step()
                lr_lowered = True
                print(
                    f"[LR] eval_mean_fC={mean_ef:.4f} > {cfg.lr_decay_fC_threshold}, 降低学习率，"
                    f"actor_lr={actor_opt.param_groups[0]['lr']:.6g}, "
                    f"critic_lr={critic_opt.param_groups[0]['lr']:.6g}"
                )
            # 若之前降低过 lr 且合作率再次跌回阈值以下，则恢复到初始 lr，允许后续再次衰减
            elif lr_lowered and (mean_ef <= cfg.lr_decay_fC_threshold):
                for pg in actor_opt.param_groups:
                    pg["lr"] = init_actor_lr
                for pg in critic_opt.param_groups:
                    pg["lr"] = init_critic_lr
                lr_lowered = False
                print(
                    f"[LR] eval_mean_fC={mean_ef:.4f} <= {cfg.lr_decay_fC_threshold}, 恢复学习率，"
                    f"actor_lr={actor_opt.param_groups[0]['lr']:.6g}, "
                    f"critic_lr={critic_opt.param_groups[0]['lr']:.6g}"
                )
            # 如表现提升，则记录为 best 并可额外保存
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

    # 训练结束后，如需要则保存模型参数
    if cfg.save_models:
        torch.save(actor.state_dict(), os.path.join(run_dir, "actor.pt"))
        torch.save(actor_target.state_dict(), os.path.join(run_dir, "actor_target.pt"))
        torch.save(critic1.state_dict(), os.path.join(run_dir, "critic1.pt"))
        torch.save(critic2.state_dict(), os.path.join(run_dir, "critic2.pt"))
        torch.save(critic1_target.state_dict(), os.path.join(run_dir, "critic1_target.pt"))
        torch.save(critic2_target.state_dict(), os.path.join(run_dir, "critic2_target.pt"))
        print(f"[TD3] Saved models to directory={run_dir}")

    # 为图像标题准备一个简短的参数描述
    title_suffix = f"L={L}, r={r}, steps={cfg.total_steps}, bs={cfg.batch_size}"

    # 训练结束后画图：episode 奖励、合作率以及 loss 曲线
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

    # eval 曲线（如果有）
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

    # 结束采样进程
    stop_event.set()
    for p in workers:
        p.join(timeout=5.0)

    if log_file is not None:
        log_file.close()
    if tb_writer is not None:
        tb_writer.close()

    if plt is not None and len(critic_loss_hist) > 0:
        xs = list(range(len(critic_loss_hist)))
        plt.figure()
        plt.plot(xs, critic_loss_hist, label="critic loss")
        if actor_loss_hist:
            xs_a = list(range(len(actor_loss_hist)))
            plt.plot(xs_a, actor_loss_hist, label="actor loss")
        plt.xlabel("update index")
        plt.ylabel("loss")
        plt.title(f"TD3: losses ({title_suffix})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(run_dir, "td3_losses.png") if cfg.save_models else "td3_losses.png"
        plt.savefig(out_path)
        plt.close()

    if log_file is not None:
        log_file.close()


def evaluate_trained_actor(
    actor_path: str,
    L: int,
    r: float,
    episode_length: int = 500,
    eval_episodes: int = 5,
    device: str = "cpu",
):
    """
    读取已经训练好的 actor，并在新的 (L, r) 下评估若干 episode。
    """
    actor = ActorNet().to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    env = PublicGoodsEnv(L=L, r=r, episode_length=episode_length, use_cumulative_planner_reward=False)
    rewards = []
    fcs = []
    for ep in range(eval_episodes):
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
        print(f"[Eval] episode {ep+1}/{eval_episodes}, reward={ep_reward:.4f}, mean_fC={fcs[-1]:.4f}")

    mean_reward = float(np.mean(rewards))
    mean_fC = float(np.mean(fcs))
    print(f"[Eval] actor={actor_path}, L={L}, r={r}, mean_reward={mean_reward:.4f}, mean_fC={mean_fC:.4f}")
    return mean_reward, mean_fC


def _evaluate_combo_worker(args):
    """
    多进程 eval 的 worker 包装：
    接收一组参数，调用 evaluate_trained_actor 并返回带 L/r/episode_length 的结果。
    """
    actor_path, L_eval, r_eval, ep_len, eval_episodes, device = args
    mean_reward, mean_fC = evaluate_trained_actor(
        actor_path=actor_path,
        L=L_eval,
        r=r_eval,
        episode_length=ep_len,
        eval_episodes=eval_episodes,
        device=device,
    )
    return L_eval, r_eval, ep_len, eval_episodes, mean_reward, mean_fC


if __name__ == "__main__":
    # 示例配置：把所有重要超参集中在 TD3Config 中，便于一处调参
    cfg1 = TD3Config(
        device="cuda:0",             # 训练设备（"cpu" / "cuda:0" 等）
        gamma=0.99,                  # 折扣因子
        actor_lr=1e-4,               # Actor 学习率
        critic_lr=1e-4,              # Critic 学习率
        tau=0.005,                   # target 网络软更新系数
        policy_noise=0.03,           # target smoothing 噪声强度
        noise_clip=0.05,             # target smoothing 噪声截断范围
        expl_noise=0.005,             # 行为策略探索噪声
        policy_delay=2,              # policy 延迟更新频率
        batch_size=1024,               # 每次更新采样的 batch 大小
        replay_size = 300_000,         # 经验回放容量
        total_steps=3_000_000,         # 总交互步数
        start_steps=50_000,          # 纯探索步数
        eval_interval=6_000,         # 评估间隔（<=0 表示不评估）
        eval_episodes = 5,           # 评估次数略增，平滑曲线
        save_models=True,            # 是否保存 checkpoint
        save_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),  # checkpoint 保存目录（绝对路径）
        load_run_id=None,            # 若要从已有 run 续训，在此填入 run_id
        save_best=True,              # eval 表现更好时是否保存 best_*.pt
        early_stop_patience=10,       # 若 >0，则 eval reward 连续若干次未提升时提前停止
        min_steps_for_early_stop=600_000,  # 提前停止前的最小训练步数
        early_stop_fC_threshold=0.7,  # eval 合作率高于该阈值且 reward 未提升才允许早停
        lr_decay_fC_threshold= 0.7,  # eval 合作率高于该阈值时触发 lr 衰减
        lr_decay_multiplier=0.25,    # lr 衰减乘子（乘在当前 lr 上）
        rollout_workers = 24,        # 并行采样进程数（默认自动）
        samples_per_step = 64  # 每个训练 step 从 data_queue 最多取多少条样本；None 默认等于 rollout_workers 或 4

    )
#继续训练 配置
    cfg2 = TD3Config(
        device="cuda:1",             # 训练设备（"cpu" / "cuda:0" 等）
        gamma=0.99,                  # 折扣因子
        actor_lr=1e-4,               # Actor 学习率
        critic_lr=1e-4,              # Critic 学习率
        tau=0.005,                   # target 网络软更新系数
        policy_noise=0.03,           # target smoothing 噪声强度
        noise_clip=0.05,             # target smoothing 噪声截断范围
        expl_noise=0.005,             # 行为策略探索噪声
        policy_delay=2,              # policy 延迟更新频率
        batch_size=512,               # 每次更新采样的 batch 大小
        replay_size=500_000,         # 经验回放容量
        total_steps=1500_000,         # 总交互步数
        start_steps=0,            # 纯探索步数 如果加载模型的话就调低一点啊啊啊   10_000
        eval_interval=6_000,         # 评估间隔（<=0 表示不评估）
        eval_episodes = 5,           # 评估次数略增，平滑曲线
        save_models=True,            # 是否保存 checkpoint
        save_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),  # checkpoint 保存目录（绝对路径）
        load_run_id=None,            # 若要从已有 run 续训，在此填入 run_id
        save_best=True,              # eval 表现更好时是否保存 best_*.pt
        early_stop_patience=15,       # 若 >0，则 eval reward 连续若干次未提升时提前停止
        min_steps_for_early_stop=600_000,  # 早停生效的最小 step，避免一开始就早停
        early_stop_fC_threshold=0.7,  # eval 合作率高于该阈值且 reward 未提升才允许早停
        lr_decay_fC_threshold= 0.7,  # eval 合作率高于该阈值时触发 lr 衰减
        lr_decay_multiplier=0.25,    # lr 衰减乘子（乘在当前 lr 上）
        rollout_workers=24,        # 并行采样进程数（默认自动）
        samples_per_step = 64  # 每个训练 step 从 data_queue 最多取多少条样本；None 默认等于 rollout_workers 或 4

    )
    # 若需要仅评估已有模型，可在这里配置
    EVAL_ONLY = False           #是否只是加载模型并且评估，不训练
    EVAL_USE_MULTIPROCESS = True      # 是否使用多进程在多个 CPU 上并行评估
    EVAL_NUM_WORKERS = 36           # 并行进程数；None 表示使用 mp.cpu_count()
    EVAL_RUN_ID = "20251119_230050第一版T3D较好效果"          # 需要评估的 run_id，例如 "20241120_153045"
    EVAL_L_LIST = [25, 30, 35, 40]                     # 可在列表中放多个 L
    EVAL_R_LIST = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]  # 可在列表中放多个 r
    EVAL_EPISODE_LENGTH_LIST = [150,50000]                   # 可在列表中放多个 episode_length
    EVAL_EPISODES = 10                                 # 每个组合评估的 episode 数

    if EVAL_ONLY:
        import multiprocessing as mp

        if not EVAL_RUN_ID:
            raise ValueError("EVAL_ONLY=True 时必须设置 EVAL_RUN_ID")
        actor_path = os.path.join(cfg1.save_dir, EVAL_RUN_ID, "actor.pt")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_csv = os.path.join(cfg1.save_dir, f"eval_results_{EVAL_RUN_ID}_{timestamp}.csv")

        combos = [
            (actor_path, L_eval, r_eval, ep_len, EVAL_EPISODES, cfg1.device)
            for L_eval in EVAL_L_LIST
            for r_eval in EVAL_R_LIST
            for ep_len in EVAL_EPISODE_LENGTH_LIST
        ]

        if EVAL_USE_MULTIPROCESS:
            num_workers = EVAL_NUM_WORKERS or mp.cpu_count()
            print(f"[Eval] 使用多进程并行评估，进程数={num_workers}，组合数={len(combos)}")
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_evaluate_combo_worker, combos)
        else:
            print(f"[Eval] 使用单进程顺序评估，组合数={len(combos)}")
            results = []
            for args in combos:
                results.append(_evaluate_combo_worker(args))

        with open(eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["L", "r", "episode_length", "eval_episodes", "mean_reward", "mean_fC"])
            for L_eval, r_eval, ep_len, eval_episodes, mean_reward, mean_fC in results:
                writer.writerow([L_eval, r_eval, ep_len, eval_episodes, mean_reward, mean_fC])

        print(f"[Eval] 保存所有组合结果到 {eval_csv}")
    else:
        train_td3(
            L=25,                        # 棋盘边长
            r=4.0,                       # 公共物品放大因子
            episode_length=150,          # 每个 episode 的最大步数
            cfg=cfg1,
            initial_R=50,                 # 初始资源
        )
