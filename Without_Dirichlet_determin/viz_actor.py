"""
可视化已训练的 TD3 Actor 输出的分配场（pi_field）以及输入状态。

功能（首版）：
- 载入指定的 actor.pt；
- 在环境上跑一段轨迹，收集若干状态 + 对应的 actor 输出；
- 为每个采样到的时刻保存一张可视化图片，包括：
  - 三个输入通道：当前可合作策略/上一轮策略/归一化公共池；
  - 五个动作通道：mid/up/down/left/right 的分配比例热力图。

用法示例（在 Without_Dirichlet_determin 目录下）：
    python viz_actor.py --actor-path checkpoints/<run_id>/actor.pt \\
        --L 25 --r 4.0 --episode-length 150 --num-states 3 --out-dir viz_outputs
"""

from __future__ import annotations

import argparse
import os
import datetime
from typing import List, Tuple

import matplotlib
# 使用无界面后台，避免 Tk pixmap 相关错误
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env import PublicGoodsEnv
from planner_net import ActorNet

# 解决中文/符号显示问题：优先用 SimHei，没有则回退到常见 sans-serif
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 便于直接在 main 中修改的默认参数（可被命令行覆盖）
# actor_path: 训练好的模型路径；其余均为环境和采样/输出控制参数。
DEFAULT_CFG = {
    "actor_path": "checkpoints/20251119_230050第一版T3D较好效果/actor.pt",         # 模型路径（必填），示例 checkpoints/<run_id>/actor.pt
    "L": 40,                  # 棋盘边长
    "r": 4.0,                 # 公共物品放大因子
    "episode_length": 150,    # rollout 时每个 episode 的最大步数
    "num_states": 150,          # 需要可视化的状态数量
    "max_steps": 300,         # 最多滚动多少步来采集 num_states
    "device": "cuda",         # 运行设备，如 "cpu" 或 "cuda:0"
    "out_dir": "viz_outputs",  # 输出图片目录
    "saliency": True,          # 是否输出敏感度相关图（总敏感度/通道敏感度/全局敏感度）
    "saliency_all": True,      # 是否对棋盘所有格点做全局敏感度（平均聚合）
    "saliency_channels": True, # 是否输出通道级敏感度图
    # 敏感度目标：关注“哪个个体（格点）”的动作输出；当为 -1 时表示“不指定单点”
    "saliency_target_row": -1,  # 行索引，>=0 时才做单点敏感度
    "saliency_target_col": -1,  # 列索引，>=0 时才做单点敏感度
    "grad_cam": True,              # 是否输出 Grad-CAM（空间注意力）
    "grad_cam_all": True,          # 是否对棋盘所有格点做 Grad-CAM（会取平均聚合，避免输出过多图片）
    "ig_channels": True,           # 是否输出通道贡献柱状图（Integrated Gradients）
    "ig_baseline": True,          # 是否用全 0 baseline 计算 IG
    "ig_trajectory_baseline": True,  # 是否用轨迹起点 state(t_initial) 作为 baseline 计算 IG
    "ig_t_initial": 0,             # IG 轨迹 baseline 的起点索引（在收集到的状态列表中的索引，默认0表示第一帧）
    "ig_t_end": 149,                # IG 终点索引（-1 表示最后一帧）
    "ig_steps_baseline": 50,       # 全0 baseline 模式下的插值步数
    "ig_steps_traj": 50,           # 轨迹 baseline 模式下的插值步数
}

# 简单清洗热力/注意力图，避免 inf/NaN 导致 imshow 归一化溢出
def _clean_heatmap(arr: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    finite = clean[np.isfinite(clean)]
    if finite.size == 0:
        return np.zeros_like(clean)
    clip_val = np.percentile(np.abs(finite), 99.9)
    if clip_val > 0:
        clean = np.clip(clean, -clip_val, clip_val)
    if clean.max() > clean.min():
        clean = (clean - clean.min()) / (clean.max() - clean.min() + 1e-8)
    else:
        clean = np.zeros_like(clean)
    return clean


def load_actor(actor_path: str, device: str) -> ActorNet:
    """加载已训练的 actor.pt，切到 eval 模式。"""
    actor = ActorNet().to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    return actor


def select_action(actor: ActorNet, state: np.ndarray, device: str) -> np.ndarray:
    """
    给定 state，输出 env.step 所需格式的 pi_field (L, L, 5)，不加噪声。
    这里沿用训练时的确定性前向，但不加任何探索扰动。
    """
    with torch.no_grad():
        s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)  # (1,3,L,L)
        pi = actor(s_tensor)  # (1,5,L,L)
        pi_np = pi.cpu().numpy()[0]  # (5,L,L)
        pi_field = np.transpose(pi_np, (1, 2, 0))  # (L,L,5)
    return pi_field


def rollout_states(
    actor: ActorNet,
    env: PublicGoodsEnv,
    device: str,
    num_states: int,
    max_steps: int,
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """
    在 env 上执行若干步，收集 num_states 个 (state, pi_field, info)。
    - 每一步都用 actor 的确定性输出；若 episode 结束则自动 reset。
    - 最多跑 max_steps，防止一直采不到足够状态。
    """
    collected = []
    state = env.reset()
    steps = 0
    while len(collected) < num_states and steps < max_steps:
        pi_field = select_action(actor, state, device)
        next_state, reward, done, info = env.step(pi_field)
        collected.append((state, pi_field, info))
        state = next_state
        steps += 1
        if done:
            state = env.reset()
    return collected


def plot_single_state(
    state: np.ndarray,
    pi_field: np.ndarray,
    info: dict,
    out_path: str,
):
    """
    保存一张图：3 个输入通道 + 5 个动作通道。
    - 上排：可合作策略、上一轮策略、归一化公共池
    - 下排：mid/up/down/left/right 分配比例（左右合并为箭头场便于直观）
    """
    stra_now = state[0]
    stra_prev = state[1]
    p_center = state[2]

    # 动作通道（假定顺序 mid, up, down, left, right）
    mid = pi_field[:, :, 0]
    up = pi_field[:, :, 1]
    down = pi_field[:, :, 2]
    left = pi_field[:, :, 3]
    right = pi_field[:, :, 4]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"State viz | f_C={info.get('f_C', 0):.3f}, avg_net={info.get('avg_net', 0):.3f}, t={info.get('t', 0)}",
        fontsize=12,
    )

    im0 = axes[0, 0].imshow(stra_now, vmin=0, vmax=1, cmap="Greens")
    axes[0, 0].set_title("Stra_now (can cooperate)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(stra_prev, vmin=0, vmax=1, cmap="Greens")
    axes[0, 1].set_title("Stra_prev")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(p_center, cmap="Blues")
    axes[0, 2].set_title("P_center_norm")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[0, 3].axis("off")
    axes[0, 3].text(0.0, 0.5, "Inputs", fontsize=12)

    im3 = axes[1, 0].imshow(mid, vmin=0, vmax=1, cmap="OrRd")
    axes[1, 0].set_title("pi mid")
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(up, vmin=0, vmax=1, cmap="OrRd")
    axes[1, 1].set_title("pi up")
    fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(down, vmin=0, vmax=1, cmap="OrRd")
    axes[1, 2].set_title("pi down")
    fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # 把左右方向放在同一 subplot，用箭头表示差分，便于直观
    ax_quiver = axes[1, 3]
    ax_quiver.set_title("pi left/right (quiver)")
    Y, X = np.mgrid[0:mid.shape[0], 0:mid.shape[1]]
    U = right - left  # x 方向箭头
    V = np.zeros_like(U)  # 只展示左右方向
    ax_quiver.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, width=0.002)
    ax_quiver.set_xlim(-0.5, mid.shape[1] - 0.5)
    ax_quiver.set_ylim(mid.shape[0] - 0.5, -0.5)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_saliency(
    actor: ActorNet,
    state: np.ndarray,
    device: str,
    target_row: int,
    target_col: int,
) -> np.ndarray:
    """
    梯度敏感度图：衡量输出对输入每个格点/通道的影响。
    说明：
    - 目标标量：某个格点 (row, col) 的 5 个方向分配概率的平均值。
    - 对输入 state 求梯度，取绝对值后在通道维求和，得到 (L, L) 的敏感度热力。
    """
    actor.eval()
    s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    s_tensor.requires_grad_(True)
    pi = actor(s_tensor)  # (1,5,L,L)

    # 取目标格点；若传入 -1 则自动取中心
    _, _, H, W = pi.shape
    row = target_row if target_row >= 0 else H // 2
    col = target_col if target_col >= 0 else W // 2
    row = max(0, min(row, H - 1))
    col = max(0, min(col, W - 1))

    # 对该格点的 5 个方向取均值，作为敏感度目标
    target = pi[:, :, row, col].mean()

    actor.zero_grad()
    target.backward()

    # 梯度绝对值在通道维求和 -> (L, L)，再做一次 0~1 归一化，便于和全局版本对齐
    sal = s_tensor.grad.abs().sum(dim=1)[0].detach().cpu().numpy()
    if sal.max() > sal.min():
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    else:
        sal = np.zeros_like(sal)
    return sal


def compute_saliency_channels(
    actor: ActorNet,
    state: np.ndarray,
    device: str,
    target_row: int,
    target_col: int,
) -> np.ndarray:
    """
    通道级敏感度（全局版）：
    - 对棋盘所有格点的动作输出做敏感度，逐点求梯度；
    - 对每个点的三个通道梯度分别做 0~1 归一化；
    - 再对所有格点的通道敏感度求平均，得到 (3, L, L) 的全局通道敏感度图。

    返回:
        sal_ch (3, L, L)，每个通道均已单独做 0~1 归一化。
    说明：
        target_row/target_col 参数保留以兼容接口，但在全局版本中不再使用。
    """
    actor.eval()
    s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    s_tensor.requires_grad_(True)
    pi = actor(s_tensor)  # (1,5,H,W)
    _, _, H, W = pi.shape

    sal_maps = []
    for row in range(H):
        for col in range(W):
            target = pi[:, :, row, col].mean()
            actor.zero_grad()
            if s_tensor.grad is not None:
                s_tensor.grad.zero_()
            target.backward(retain_graph=True)
            grad = s_tensor.grad.detach().cpu().numpy()[0]  # (3,H,W)
            ch_maps = []
            for c in range(grad.shape[0]):
                sal = np.abs(grad[c])
                if sal.max() > sal.min():
                    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
                else:
                    sal = np.zeros_like(sal)
                ch_maps.append(sal)
            sal_maps.append(np.stack(ch_maps, axis=0))  # (3,H,W)

    if not sal_maps:
        return None
    sal_avg = np.mean(sal_maps, axis=0)  # (3,H,W)
    # 再对每个通道做一次归一化，保证 0~1
    for c in range(sal_avg.shape[0]):
        ch = sal_avg[c]
        if ch.max() > ch.min():
            sal_avg[c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        else:
            sal_avg[c] = np.zeros_like(ch)
    return sal_avg  # (3,L,L)


def compute_saliency_all(
    actor: ActorNet,
    state: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    对棋盘所有格点的动作输出做敏感度，逐点求梯度后平均，得到一张全局敏感度图。
    计算量约为 L*L 次梯度；L 较大时请谨慎开启。
    """
    actor.eval()
    s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    s_tensor.requires_grad_(True)
    pi = actor(s_tensor)  # (1,5,H,W)
    _, _, H, W = pi.shape

    sal_maps = []
    for row in range(H):
        for col in range(W):
            target = pi[:, :, row, col].mean()
            actor.zero_grad()
            if s_tensor.grad is not None:
                s_tensor.grad.zero_()
            target.backward(retain_graph=True)
            sal = s_tensor.grad.abs().sum(dim=1)[0].detach().cpu().numpy()  # (H,W)
            if sal.max() > sal.min():
                sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
            else:
                sal = np.zeros_like(sal)
            sal_maps.append(sal)

    if not sal_maps:
        return None
    sal_avg = np.mean(sal_maps, axis=0)
    if sal_avg.max() > sal_avg.min():
        sal_avg = (sal_avg - sal_avg.min()) / (sal_avg.max() - sal_avg.min() + 1e-8)
    else:
        sal_avg = np.zeros_like(sal_avg)
    return sal_avg


def compute_grad_cam(
    actor: ActorNet,
    state: np.ndarray,
    device: str,
    target_row: int,
    target_col: int,
) -> np.ndarray:
    """
    简化版 Grad-CAM：查看目标格点动作对卷积特征的空间注意力。
    做法：
    - 把 ConvBody 的最后一层卷积输出作为特征图，直接用 autograd.grad 求梯度；
    - 目标标量：目标格点 (row,col) 的 5 个方向分配概率均值；
    - 权重 = 梯度在空间上的平均，CAM = ReLU( sum_k weight_k * feature_k )，再归一化到 [0,1]。
    返回：cam (H, W)，同输入空间大小。
    """
    actor.eval()

    # 前向时记录目标卷积层输出（不做 detach，便于后续求导）
    feature_ref = {}

    def fwd_hook(module, inp, output):
        feature_ref["tensor"] = output

    target_layer = actor.body.net[-2]  # 最后一层 Conv
    handle_fwd = target_layer.register_forward_hook(fwd_hook)

    s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    s_tensor.requires_grad_(True)
    pi = actor(s_tensor)  # (1,5,H,W)

    _, _, H, W = pi.shape
    row = target_row if target_row >= 0 else H // 2
    col = target_col if target_col >= 0 else W // 2
    row = max(0, min(row, H - 1))
    col = max(0, min(col, W - 1))

    target = pi[:, :, row, col].mean()
    actor.zero_grad()

    feature = feature_ref.get("tensor")
    if feature is None:
        handle_fwd.remove()
        return None

    grad = torch.autograd.grad(outputs=target, inputs=feature, retain_graph=False, allow_unused=True)[0]

    handle_fwd.remove()

    if grad is None:
        return None

    weights = grad.mean(dim=(2, 3), keepdim=True)            # (1,C,1,1)
    cam = (weights * feature).sum(dim=1, keepdim=True)       # (1,1,H,W)
    cam = torch.relu(cam)
    cam = cam.squeeze().detach().cpu().numpy()               # (H,W)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = np.zeros_like(cam)  # 梯度全零时直接返回 0 图，避免色条显示奇怪范围
    return cam


def compute_grad_cam_all(
    actor: ActorNet,
    state: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    对棋盘所有格点分别做 Grad-CAM（目标为该格点 5 向分配均值），再求平均，得到一张全局注意力图。
    说明：
    - 避免输出 L*L 张图，改为把所有格点的 Grad-CAM 取均值，作为“全局关注”热力。
    - 计算量约为 L*L 次 autograd.grad，L 不宜过大（默认 L=25 还可接受）。
    """
    actor.eval()
    feature_ref = {}

    def fwd_hook(module, inp, output):
        feature_ref["tensor"] = output

    target_layer = actor.body.net[-2]
    handle_fwd = target_layer.register_forward_hook(fwd_hook)

    s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
    s_tensor.requires_grad_(True)
    pi = actor(s_tensor)  # (1,5,H,W)

    _, _, H, W = pi.shape
    feature = feature_ref.get("tensor")
    handle_fwd.remove()
    if feature is None:
        return None

    cams = []
    for row in range(H):
        for col in range(W):
            target = pi[:, :, row, col].mean()
            grad = torch.autograd.grad(outputs=target, inputs=feature, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                continue
            weights = grad.mean(dim=(2, 3), keepdim=True)            # (1,C,1,1)
            cam = (weights * feature).sum(dim=1, keepdim=True)       # (1,1,H,W)
            cam = torch.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()               # (H,W)
            if cam.max() > cam.min():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)
            cams.append(cam)

    if not cams:
        return None
    cam_avg = np.mean(cams, axis=0)
    if cam_avg.max() > cam_avg.min():
        cam_avg = (cam_avg - cam_avg.min()) / (cam_avg.max() - cam_avg.min() + 1e-8)
    else:
        cam_avg = np.zeros_like(cam_avg)
    return cam_avg


def plot_saliency(
    state: np.ndarray,
    saliency: np.ndarray,
    out_path: str,
    title: str,
    L: int,
    r: float,
    t: int,
):
    """
    将敏感度热力图与输入状态并排展示，结构与 Grad-CAM 保持一致：
    - 本轮可合作策略 / 上一轮策略 / P_center_norm / 全局或单点敏感度。
    """
    stra_now = state[0]
    stra_prev = state[1]
    p_center = state[2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{title} | t={t}, L={L}, r={r}", fontsize=12)

    im0 = axes[0].imshow(stra_now, vmin=0, vmax=1, cmap="Greens")
    axes[0].set_title("Stra_now (本轮可合作)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im_prev = axes[1].imshow(stra_prev, vmin=0, vmax=1, cmap="Greens")
    axes[1].set_title("Stra_prev (上一轮策略)")
    fig.colorbar(im_prev, ax=axes[1], fraction=0.046, pad=0.04)

    im1 = axes[2].imshow(p_center, cmap="Blues")
    axes[2].set_title("P_center_norm")
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

    saliency_clean = _clean_heatmap(saliency)
    im2 = axes[3].imshow(saliency_clean, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Saliency (敏感度)")
    fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_saliency_channels(
    state: np.ndarray,
    saliency_ch: np.ndarray,
    out_path: str,
    title: str,
    L: int,
    r: float,
    t: int,
):
    """
    通道级敏感度可视化：
    - 通道 0: 上一轮策略 Stra_prev
    - 通道 1: 当前可合作策略 Stra_now
    - 通道 2: 归一化公共池 P_center_norm
    每个子图展示对应通道的敏感度热力图。
    """
    stra_prev = state[1]
    stra_now = state[0]
    p_center = state[2]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{title} (per-channel) | t={t}, L={L}, r={r}", fontsize=12)

    # 上一轮策略 + 敏感度
    im_prev = axes[0, 0].imshow(stra_prev, vmin=0, vmax=1, cmap="Greens")
    axes[0, 0].set_title("Stra_prev (上一轮策略)")
    fig.colorbar(im_prev, ax=axes[0, 0], fraction=0.046, pad=0.04)

    sal_prev_clean = _clean_heatmap(saliency_ch[0])
    im_prev_sal = axes[1, 0].imshow(sal_prev_clean, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Saliency: 通道1 (上一轮)")
    fig.colorbar(im_prev_sal, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 当前可合作策略 + 敏感度
    im_now = axes[0, 1].imshow(stra_now, vmin=0, vmax=1, cmap="Greens")
    axes[0, 1].set_title("Stra_now (本轮可合作)")
    fig.colorbar(im_now, ax=axes[0, 1], fraction=0.046, pad=0.04)

    sal_now_clean = _clean_heatmap(saliency_ch[1])
    im_now_sal = axes[1, 1].imshow(sal_now_clean, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("Saliency: 通道2 (本轮)")
    fig.colorbar(im_now_sal, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 公共池 + 敏感度
    im_p = axes[0, 2].imshow(p_center, cmap="Blues")
    axes[0, 2].set_title("P_center_norm")
    fig.colorbar(im_p, ax=axes[0, 2], fraction=0.046, pad=0.04)

    sal_p_clean = _clean_heatmap(saliency_ch[2])
    im_p_sal = axes[1, 2].imshow(sal_p_clean, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1, 2].set_title("Saliency: 通道3 (资源)")
    fig.colorbar(im_p_sal, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_ig_channels(
    actor: ActorNet,
    baseline_state: np.ndarray,
    target_state: np.ndarray,
    device: str,
    m_steps: int = 20,
) -> np.ndarray:
    """
    通道级 Integrated Gradients（全局版）：
    - baseline 取全 0；
    - 从 baseline 到 state 做 m_steps 次线性插值；
    - 每一步以全局动作输出 pi.mean() 为目标标量；
    - 累积梯度近似积分，得到每个通道的 IG；
    - 在空间上求和，得到 3 个通道的整体贡献，并做绝对值归一化到和为 1。

    返回:
        contrib (3,)  对应 [上一轮策略, 当前策略, 公共池] 的相对贡献。
    """
    actor.eval()

    state_np = target_state.astype(np.float32)
    baseline_np = baseline_state.astype(np.float32)

    state_t = torch.from_numpy(state_np).unsqueeze(0).to(device)    # (1,3,L,L)
    baseline_t = torch.from_numpy(baseline_np).unsqueeze(0).to(device)

    total_grad = torch.zeros_like(state_t)

    for k in range(1, m_steps + 1):
        alpha = float(k) / float(m_steps)
        x = baseline_t + alpha * (state_t - baseline_t)
        x.requires_grad_(True)
        pi = actor(x)  # (1,5,L,L)
        target = pi.mean()  # 全局动作输出的平均值

        actor.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        target.backward()
        total_grad += x.grad

    avg_grad = total_grad / float(m_steps)  # (1,3,L,L)
    ig = (state_t - baseline_t) * avg_grad  # (1,3,L,L)
    ig_np = ig.detach().cpu().numpy()[0]    # (3,L,L)

    contrib = ig_np.reshape(3, -1).sum(axis=1)  # 每个通道在空间上的总贡献
    contrib_abs = np.abs(contrib)
    if contrib_abs.sum() > 0:
        contrib_norm = contrib_abs / (contrib_abs.sum() + 1e-8)
    else:
        contrib_norm = np.zeros_like(contrib_abs)
    return contrib_norm  # (3,)


def plot_ig_bar(
    contrib: np.ndarray,
    out_path: str,
    title: str,
    L: int,
    r: float,
    t: int,
):
    """
    绘制通道贡献柱状图：
    - x 轴：三个通道（上一轮 / 当前 / 公共池）
    - y 轴：Integrated Gradients 归一化贡献（0~1，绝对值归一）
    """
    labels = ["通道1(上一轮)", "通道2(本轮)", "通道3(资源)"]
    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar(x, contrib, color=["#4e79a7", "#f28e2c", "#e15759"])
    plt.xticks(x, labels, rotation=20)
    plt.ylim(0.0, 1.0)
    plt.ylabel("归一化通道贡献")
    plt.title(f"{title} | t={t}, L={L}, r={r}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_grad_cam(
    state: np.ndarray,
    cam: np.ndarray,
    out_path: str,
    title: str,
    L: int,
    r: float,
    t: int,
):
    """
    将 Grad-CAM 热力图与输入状态并排展示，便于看“空间注意力”。
    同时展示上一轮/当前策略，避免仅看单轮。
    """
    stra_now = state[0]
    stra_prev = state[1]
    p_center = state[2]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"{title} | t={t}, L={L}, r={r}", fontsize=12)

    im0 = axes[0].imshow(stra_now, vmin=0, vmax=1, cmap="Greens")
    axes[0].set_title("Stra_now (本轮可合作)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im_prev = axes[1].imshow(stra_prev, vmin=0, vmax=1, cmap="Greens")
    axes[1].set_title("Stra_prev (上一轮策略)")
    fig.colorbar(im_prev, ax=axes[1], fraction=0.046, pad=0.04)

    im1 = axes[2].imshow(p_center, cmap="Blues")
    axes[2].set_title("P_center_norm")
    fig.colorbar(im1, ax=axes[2], fraction=0.046, pad=0.04)

    cam_clean = _clean_heatmap(cam)
    im2 = axes[3].imshow(cam_clean, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Grad-CAM (空间注意力)")
    fig.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize trained TD3 actor outputs on sampled states.")
    parser.add_argument("--actor-path", type=str, default=DEFAULT_CFG["actor_path"], help="actor.pt 路径（必填）")
    parser.add_argument("--L", type=int, default=DEFAULT_CFG["L"], help="棋盘边长 L")
    parser.add_argument("--r", type=float, default=DEFAULT_CFG["r"], help="公共物品放大因子 r")
    parser.add_argument("--episode-length", type=int, default=DEFAULT_CFG["episode_length"], help="rollout 的 episode 最大步数")
    parser.add_argument("--num-states", type=int, default=DEFAULT_CFG["num_states"], help="要可视化的状态数量")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_CFG["max_steps"], help="最多滚动多少步来收集状态")
    parser.add_argument("--device", type=str, default=DEFAULT_CFG["device"], help="运行设备，如 cpu 或 cuda:0")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_CFG["out_dir"], help="输出图片目录（根目录，将在其中创建子目录）")
    parser.add_argument("--saliency", action="store_true", default=DEFAULT_CFG["saliency"], help="是否输出敏感度相关图")
    parser.add_argument(
        "--saliency-all",
        action="store_true",
        default=DEFAULT_CFG["saliency_all"],
        help="是否对棋盘所有格点做敏感度并求平均（全局敏感度）",
    )
    parser.add_argument(
        "--saliency-target-row",
        type=int,
        default=DEFAULT_CFG["saliency_target_row"],
        help="敏感度关注的行（-1 表示自动取中心）",
    )
    parser.add_argument(
        "--saliency-target-col",
        type=int,
        default=DEFAULT_CFG["saliency_target_col"],
        help="敏感度关注的列（-1 表示自动取中心）",
    )
    parser.add_argument(
        "--saliency-channels",
        action="store_true",
        default=DEFAULT_CFG["saliency_channels"],
        help="是否输出通道级敏感度图（仅在指定单点时有效）",
    )
    parser.add_argument("--grad-cam", action="store_true", default=DEFAULT_CFG["grad_cam"], help="是否输出 Grad-CAM 图")
    parser.add_argument(
        "--grad-cam-all",
        action="store_true",
        default=DEFAULT_CFG["grad_cam_all"],
        help="是否对棋盘所有格点做 Grad-CAM 并求均值（全局关注图）",
    )
    parser.add_argument(
        "--ig-channels",
        action="store_true",
        default=DEFAULT_CFG["ig_channels"],
        help="是否输出通道贡献柱状图（Integrated Gradients）",
    )
    parser.add_argument(
        "--ig-baseline",
        action="store_true",
        default=DEFAULT_CFG["ig_baseline"],
        help="是否使用全 0 baseline 计算 IG 通道贡献",
    )
    parser.add_argument(
        "--ig-trajectory-baseline",
        action="store_true",
        default=DEFAULT_CFG["ig_trajectory_baseline"],
        help="是否使用轨迹起点 state(t_initial) 作为 baseline 计算 IG 通道贡献",
    )
    parser.add_argument(
        "--ig-t-initial",
        type=int,
        default=DEFAULT_CFG["ig_t_initial"],
        help="IG 轨迹 baseline 起点在收集状态列表中的索引（0 表示第一帧，-1 无效）",
    )
    parser.add_argument(
        "--ig-t-end",
        type=int,
        default=DEFAULT_CFG["ig_t_end"],
        help="IG 终点在收集状态列表中的索引（-1 表示最后一帧）",
    )
    parser.add_argument(
        "--ig-steps-baseline",
        type=int,
        default=DEFAULT_CFG["ig_steps_baseline"],
        help="全 0 baseline 模式下 IG 的插值步数",
    )
    parser.add_argument(
        "--ig-steps-traj",
        type=int,
        default=DEFAULT_CFG["ig_steps_traj"],
        help="轨迹 baseline 模式下 IG 的插值步数",
    )
    args = parser.parse_args()

    # 如果没有在命令行或 DEFAULT_CFG 中提供 actor_path，则给出明确提示
    if not args.actor_path:
        raise ValueError("请在 DEFAULT_CFG['actor_path'] 或命令行 --actor-path 中指定 actor.pt 路径")

    # 构造本次运行的子目录名，包含时间戳和关键参数，避免覆盖
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    subdir_name = f"L{args.L}_r{args.r}_ep{args.episode_length}_ns{args.num_states}_{timestamp}"
    save_dir = os.path.join(args.out_dir, subdir_name)
    os.makedirs(save_dir, exist_ok=True)

    # 初始化模型、环境
    actor = load_actor(args.actor_path, args.device)
    env = PublicGoodsEnv(
        L=args.L,
        r=args.r,
        episode_length=args.episode_length,
        use_cumulative_planner_reward=False,
    )

    # 滚动收集若干状态+动作，用于可视化
    collected = rollout_states(
        actor=actor,
        env=env,
        device=args.device,
        num_states=args.num_states,
        max_steps=args.max_steps,
    )

    # 逐个状态生成图片
    for idx, (state, pi_field, info) in enumerate(collected):
        out_path = os.path.join(save_dir, f"viz_state_{timestamp}_{idx}.png")
        plot_single_state(state, pi_field, info, out_path)
        print(f"[viz] saved {out_path}")

        # 可选：生成梯度敏感度图，展示“动作对输入的敏感度”
        # ====== 敏感度相关可视化 ======
        # 1) 全局平均敏感度（对所有格点求平均）
        if args.saliency:
            # 1) 全局平均敏感度（对所有格点求平均）
            if args.saliency_all:
                sal_global = compute_saliency_all(
                    actor=actor,
                    state=state,
                    device=args.device,
                )
                if sal_global is not None:
                    sal_path = os.path.join(
                        save_dir, f"viz_saliency_global_{timestamp}_{idx}.png"
                    )
                    sal_title = "Saliency (全局平均，关注所有格点)"
                    plot_saliency(
                        state,
                        sal_global,
                        sal_path,
                        sal_title,
                        L=args.L,
                        r=args.r,
                        t=info.get("t", 0),
                    )
                    print(f"[viz] saved {sal_path}")

            # 2) 单点敏感度（仅当显式指定了行列索引时）
            if args.saliency_target_row >= 0 and args.saliency_target_col >= 0:
                sal_single = compute_saliency(
                    actor=actor,
                    state=state,
                    device=args.device,
                    target_row=args.saliency_target_row,
                    target_col=args.saliency_target_col,
                )
                sal_path = os.path.join(
                    save_dir,
                    f"viz_saliency_point_{timestamp}_{idx}_r{args.saliency_target_row}_c{args.saliency_target_col}.png",
                )
                sal_title = (
                    f"Saliency @ row={args.saliency_target_row}, col={args.saliency_target_col}"
                )
                plot_saliency(
                    state,
                    sal_single,
                    sal_path,
                    sal_title,
                    L=args.L,
                    r=args.r,
                    t=info.get("t", 0),
                )
                print(f"[viz] saved {sal_path}")

        # 3) 通道级敏感度（独立开关，只要 saliency_channels=True 就会输出；
        #    若未指定单点，则默认取中心格点）
        if args.saliency_channels:
            sal_ch = compute_saliency_channels(
                actor=actor,
                state=state,
                device=args.device,
                target_row=args.saliency_target_row,
                target_col=args.saliency_target_col,
            )
            sal_ch_path = os.path.join(
                save_dir,
                f"viz_saliency_channels_{timestamp}_{idx}_r{args.saliency_target_row}_c{args.saliency_target_col}.png",
            )
            sal_ch_title = (
                f"Channel-wise Saliency @ row={args.saliency_target_row}, col={args.saliency_target_col}"
            )
            plot_saliency_channels(
                state,
                sal_ch,
                sal_ch_path,
                sal_ch_title,
                L=args.L,
                r=args.r,
                t=info.get("t", 0),
            )
            print(f"[viz] saved {sal_ch_path}")

        # 4) Grad-CAM：为每个采样状态都输出一张
        if args.grad_cam:
            if args.grad_cam_all:
                cam = compute_grad_cam_all(
                    actor=actor,
                    state=state,
                    device=args.device,
                )
                cam_title = "Grad-CAM (全局平均，关注所有格点)"
                cam_suffix = "all"
            else:
                cam = compute_grad_cam(
                    actor=actor,
                    state=state,
                    device=args.device,
                    target_row=args.saliency_target_row,
                    target_col=args.saliency_target_col,
                )
                cam_title = f"Grad-CAM @ row={args.saliency_target_row}, col={args.saliency_target_col}"
                cam_suffix = f"r{args.saliency_target_row}_c{args.saliency_target_col}"

            if cam is not None:
                cam_path = os.path.join(save_dir, f"viz_gradcam_{timestamp}_{idx}_{cam_suffix}.png")
                plot_grad_cam(state, cam, cam_path, cam_title, L=args.L, r=args.r, t=info.get("t", 0))
                print(f"[viz] saved {cam_path}")

    # 5) 通道贡献柱状图（Integrated Gradients），基于可调的 t_initial / t_end
    if args.ig_channels and collected:
        n = len(collected)
        # 终点索引：默认 -1 表示最后一帧
        if args.ig_t_end < 0:
            end_idx = n - 1
        else:
            end_idx = max(0, min(args.ig_t_end, n - 1))

        # 起点索引：默认 0 表示第一帧
        if args.ig_t_initial < 0:
            init_idx = 0
        else:
            init_idx = max(0, min(args.ig_t_initial, n - 1))

        target_state, _, target_info = collected[end_idx]

        # baseline 1: 全 0
        if args.ig_baseline:
            baseline_zero = np.zeros_like(target_state, dtype=np.float32)
            ig_contrib_zero = compute_ig_channels(
                actor=actor,
                baseline_state=baseline_zero,
                target_state=target_state,
                device=args.device,
                m_steps=args.ig_steps_baseline,
            )
            ig_path_zero = os.path.join(
                save_dir,
                f"viz_ig_channels_baseline_{timestamp}_init0_end{end_idx}.png",
            )
            ig_title_zero = "IG 通道贡献 (baseline=全0)"
            plot_ig_bar(
                ig_contrib_zero,
                ig_path_zero,
                ig_title_zero,
                L=args.L,
                r=args.r,
                t=target_info.get("t", 0),
            )
            print(f"[viz] saved {ig_path_zero}")

        # baseline 2: 轨迹起点 t_initial（同一条 rollout 的第 init_idx 个状态）
        if args.ig_trajectory_baseline and n > 1:
            baseline_state, _, baseline_info = collected[init_idx]
            ig_contrib_traj = compute_ig_channels(
                actor=actor,
                baseline_state=baseline_state,
                target_state=target_state,
                device=args.device,
                m_steps=args.ig_steps_traj,
            )
            ig_path_traj = os.path.join(
                save_dir,
                f"viz_ig_channels_traj_{timestamp}_init{init_idx}_end{end_idx}.png",
            )
            ig_title_traj = (
                f"IG 通道贡献 (baseline=t_initial, t0={baseline_info.get('t', 0)})"
            )
            plot_ig_bar(
                ig_contrib_traj,
                ig_path_traj,
                ig_title_traj,
                L=args.L,
                r=args.r,
                t=target_info.get("t", 0),
            )
            print(f"[viz] saved {ig_path_traj}")

    if not collected:
        print("[viz] no states collected, consider increasing max-steps.")


if __name__ == "__main__":
    main()
