"""
worker_utils.py

本模块包含用于 actor-learner 框架的几种核心损失与工具函数：
- V-trace（IMPALA 风格）的实现（vtrace_from_log_probs、compute_vtrace_loss）；
- 以及兼容的 A3C 损失函数 compute_a3c_loss（保留作参考，不参与当前训练）。

核心输入 / 输出张量的形状（单条轨迹）：
- target_log_probs: Tensor (T,)
  - Learner 在“当前策略”下重新计算的 logπ(a_t | s_t)；
- behavior_log_probs: Tensor (T,)
  - Actor 在采样时使用的“行为策略” logμ(a_t | s_t)，由 worker.py 记录；
- values: Tensor (T+1,)
  - V(s_0..s_T)，包含 bootstrap 的 V(s_T)；
- rewards: Tensor (T,)
  - 每步的标量 reward（本项目中来自 env.step 的 planner_reward）；
- dones: Tensor (T,)
  - 标识 episode 是否终止（未终止则为 0）。

V-trace 的核心思想：
- off-policy 情况下，behavior policy μ 与当前 target policy π 不一致；
- 通过重要性采样比率 ρ_t = exp(logπ - logμ) 做校正；
- 为了稳定训练，对 ρ_t 和 c_t = clip(ρ_t, max=c_bar) 做截断；
- 对 value function 使用“修正后的目标 vs”，对 policy 使用“校正后的优势 advantages”。

返回值：
- compute_vtrace_loss 返回 (loss_tensor, policy_loss_scalar, value_loss_scalar)，
  方便在训练循环中打印标量 loss。
"""

import torch
import torch.nn.functional as F


def vtrace_from_log_probs(
    target_log_probs,    # (T,)  learner 当前策略下 log π(a_t|s_t)
    behavior_log_probs,  # (T,)  actor 当时策略下 log μ(a_t|s_t)
    values,              # (T+1,) V(s_t)，含 bootstrap V(s_T)
    rewards,             # (T,)  r_t
    dones,               # (T,)  0/1
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
):
    """
    单条轨迹版 V-trace 算法实现（参考 IMPALA）

    算法要点：
    - 计算重要性比率 rho_t = exp(logπ - logμ)，并进行截断 rho_clipped = clip(rho_t, max=rho_bar)
    - 计算辅助截断 c_t = clip(rho_t, max=c_bar)
    - 从后向前递推得到校正后的 v_s 值

    返回：vs (T,) 对应修正后的 state 值，advantages (T,) 对应用于 policy gradient 的优势
    """
    device = values.device
    T = rewards.shape[0]

    # 重要性比率 rho, c
    log_rhos = target_log_probs - behavior_log_probs      # (T,)
    rhos = torch.exp(log_rhos)                            # (T,)
    rhos_clipped = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)

    vs = torch.zeros(T, device=device)
    # 从最后一步开始向前递推 v_s
    # v_s[T] = V(s_T)（bootstrap）
    vs_plus_1 = values[-1]  # V(s_T)

    for t in reversed(range(T)):
        # δ_t = ρ̄_t * (r_t + γ V(s_{t+1}) - V(s_t))
        delta = rhos_clipped[t] * (rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t])
        # v_s = V(s_t) + δ_t + γ c_t (v_{s+1} - V(s_{t+1}))
        vs[t] = values[t] + delta + gamma * cs[t] * (1 - dones[t]) * (vs_plus_1 - values[t + 1])
        vs_plus_1 = vs[t]

    # 对应 policy gradient 的 advantage:
    #   A_t = ρ̄_t * (r_t + γ v_{t+1} - V(s_t))
    advantages = rhos_clipped * (
        rewards + gamma * torch.cat([vs[1:], values[-1:].detach()]) * (1 - dones) - values[:-1]
    )

    return vs, advantages


def compute_vtrace_loss(
    target_log_probs,    # (T,)
    behavior_log_probs,  # (T,)
    values,              # (T+1,)
    rewards,             # list/np/torch (T,)
    dones,               # list/np/torch (T,)
    entropies=None,      # (T,)
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
    entropy_beta=0.01,
):
    """
    计算 V-trace 损失并返回 (loss, policy_loss, value_loss)

    policy_loss: 基于 advantages 与 target_log_probs
    value_loss: MSE between values[:-1] and vs (V-trace 目标)
    entropy: 可选的熵正则项（支持传入 Tensor 或 list/tuple）
    """
    device = values.device
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

    vs, advantages = vtrace_from_log_probs(
        target_log_probs=target_log_probs,
        behavior_log_probs=behavior_log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        gamma=gamma,
        rho_bar=rho_bar,
        c_bar=c_bar,
    )

    # policy loss: -E[ A_t * logπ_target ]
    policy_loss = -(advantages.detach() * target_log_probs).mean()

    # value loss: MSE between learner value estimates and vtrace targets
    value_loss = F.mse_loss(values[:-1], vs.detach())

    loss = policy_loss + 0.5 * value_loss

    # 熵正则项：支持多种传入格式（Tensor 或 list/tuple）
    if entropies is not None:
        if isinstance(entropies, (list, tuple)):
            entropies = torch.stack(entropies)
        else:
            entropies = torch.as_tensor(entropies, device=device)
        entropy_loss = -entropies.mean()
        loss = loss + entropy_beta * entropy_loss

    return loss, policy_loss.item(), value_loss.item()



def compute_a3c_loss(
    log_probs,      # list[T] of tensor: 每步的 logπ(a_t|s_t)
    values,         # list[T+1] of tensor: V(s_t)，含 bootstrap 的 V(s_{T})
    rewards,        # list[T] of float/tensor: env 给的 reward_t (标量)
    dones,          # list[T] of bool: 是否终止
    gamma=0.99,
    tau=1.0,        # GAE 平滑系数, 先用 tau=1 就是 n-step return
    entropy_terms=None,   # list[T] of tensor: 每步策略熵（可选）
    entropy_beta=0.01,
):
    """
    典型 A3C / A2C 损失（兼容旧实现）

    说明：本项目已经在使用 V-trace；A3C 作为备选/参考实现保留。
    """
    T = len(rewards)
    device = values[0].device

    # 把 reward / done 转 tensor
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    # 计算返回 G_t（从后往前）
    returns = torch.zeros(T, dtype=torch.float32, device=device)
    R = values[-1].detach()  # bootstrap: V(s_T)
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R

    values_t = torch.stack(values[:-1])          # (T,)
    log_probs_t = torch.stack(log_probs)        # (T,)

    # 优势函数 A_t = G_t - V(s_t)
    advantages = returns - values_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 策略损失: - E[ logπ(a_t|s_t) * A_t ]
    policy_loss = -(log_probs_t * advantages.detach()).mean()

    # 价值损失: MSE( V(s_t), G_t )
    value_loss = F.mse_loss(values_t, returns.detach())

    # 熵正则（鼓励探索）
    if entropy_terms is not None:
        entropies = torch.stack(entropy_terms)  # (T,)
        entropy_loss = -entropies.mean()
        loss = policy_loss + 0.5 * value_loss + entropy_beta * entropy_loss
    else:
        loss = policy_loss + 0.5 * value_loss

    return loss, policy_loss.item(), value_loss.item()
