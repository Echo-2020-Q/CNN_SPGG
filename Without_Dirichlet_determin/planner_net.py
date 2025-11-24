"""
TD3 networks for the spatial public-goods game.

本文件提供两个核心网络：
- ActorNet:  输入 state (B, 3, L, L)，输出确定性的分配场 pi (B, 5, L, L)，
            每个格点 5 维经过 softmax 后是合法的分配比例；
- CriticNet: 输入 state 和 action（pi_field），输出标量 Q(s, a)。

网络结构尽量沿用原来的 PlannerNet 卷积干路，以便和之前的实验可比。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBody(nn.Module):
    """
    共享的卷积特征提取干路。

    这里简单使用三层 3x3 卷积 + ReLU，不改变空间尺寸，
    适合在棋盘上抽取局部交互特征。
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorNet(nn.Module):
    """
    确定性 Actor：
    - 输入: state (B, 3, L, L)
    - 输出: pi_field (B, 5, L, L)，每个格点 5 维经过 softmax 后为合法概率向量。

    注意：Actor 本身是确定性的；训练和行为探索通过在 logits 上添加噪声实现，
    具体逻辑在 global_trainer.py 的 select_action 中实现。
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        # 主干卷积提取空间特征
        self.body = ConvBody(in_channels, base_channels)
        # 通过 1x1 卷积把通道压到 5，对应每个格点的 5 维分配比例 logits
        self.policy_head = nn.Conv2d(base_channels, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回确定性策略 pi_field，形状 (B, 5, L, L)，每个格点沿通道维度 softmax。
        """
        feat = self.body(x)              # (B, base, L, L)
        logits = self.policy_head(feat)  # (B, 5, L, L)
        pi = F.softmax(logits, dim=1)    # 每个格点的 5 通道归一化为概率
        return pi


class CriticNet(nn.Module):
    """
    单个 Q 网络：
    - 输入:
        state: (B, 3, L, L)
        action: (B, 5, L, L)  对应每格点的分配比例
    - 输出:
        q: (B,)  标量 Q 值

    思路：把 state 和 action 在通道维 concat 成 (B, 8, L, L)，
    经过卷积 + 全局池化 + MLP 得到一个标量 Q(s,a)。
    """

    def __init__(self, state_channels: int = 3, action_channels: int = 5, base_channels: int = 32):
        super().__init__()
        in_channels = state_channels + action_channels
        self.body = ConvBody(in_channels, base_channels)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state:  (B, 3, L, L)
        action: (B, 5, L, L)
        """
        # 在通道维拼接状态和动作，形成一个“联合输入”
        x = torch.cat([state, action], dim=1)  # (B, 8, L, L)
        feat = self.body(x)
        q = self.head(feat).squeeze(-1)  # (B,)
        return q


__all__ = ["ActorNet", "CriticNet"]
