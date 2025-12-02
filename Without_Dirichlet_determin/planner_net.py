"""
TD3 networks for the spatial public-goods game.

本文件提供两个核心网络：
- ActorNet:  输入 state (B, C, L, L)，输出确定性的分配场 pi (B, 5, L, L)，
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

    三层 3x3 卷积 + ReLU，其中后两层使用空洞卷积扩大感受野；
    搭配 1x1 残差支路，保持输出尺寸和参数量稳定。
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()
        C = base_channels
        self.conv1 = nn.Conv2d(in_channels, C, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(C, C, kernel_size=3, padding=4, dilation=4)
        self.act = nn.ReLU(inplace=True)
        # 残差支路：当输入通道与 base_channels 不一致时，用 1x1 卷积对齐
        self.skip = nn.Conv2d(in_channels, C, kernel_size=1) if in_channels != C else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.act(self.conv3(out))
        return out + self.skip(x)


class ActorNet(nn.Module):
    """
    确定性 Actor：
    - 输入: state (B, in_channels, L, L)
    - 输出: pi_field (B, 5, L, L)，每个格点 5 维经过 softmax 后为合法概率向量。

    注意：Actor 本身是确定性的；训练和行为探索通过在 logits 上添加噪声实现，
    具体逻辑在 global_trainer.py 的 select_action 中实现。
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 64):
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
        state: (B, state_channels, L, L)
        action: (B, 5, L, L)  对应每格点的分配比例
    - 输出:
        q: (B,)  标量 Q 值

    思路：把 state 和 action 在通道维 concat 成 (B, 8, L, L)，
    经过卷积 + 全局池化 + MLP 得到一个标量 Q(s,a)。
    """

    def __init__(self, state_channels: int = 4, action_channels: int = 5, base_channels: int = 64):
        super().__init__()
        in_channels = state_channels + action_channels
        self.body = ConvBody(in_channels, base_channels)
        # 卷积特征全局池化，用于提取全局分布特征
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 额外手工全局特征数量（全局 f_C、全局平均 R_norm）
        self.num_global_feats = 2
        fc_in_dim = base_channels + self.num_global_feats

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        state:  (B, state_channels, L, L)
        action: (B, 5, L, L)
        """
        # 在通道维拼接状态和动作，形成一个“联合输入”
        x = torch.cat([state, action], dim=1)  # (B, 8, L, L)
        feat = self.body(x)  # (B, base, L, L)

        # 卷积特征全局平均池化 -> (B, base)
        pooled = self.global_pool(feat).view(feat.size(0), -1)

        # 手工全局特征：全局可合作率（通道0均值）、全局平均 R_norm（通道3均值）
        f_c_global = state[:, 0].mean(dim=(1, 2))
        rnorm_global = state[:, 3].mean(dim=(1, 2))
        global_feats = torch.stack([f_c_global, rnorm_global], dim=1)  # (B, 2)

        # 拼接卷积全局特征 + 手工全局特征
        h = torch.cat([pooled, global_feats], dim=1)  # (B, base+2)

        q = self.head(h).squeeze(-1)  # (B,)
        return q


__all__ = ["ActorNet", "CriticNet"]
