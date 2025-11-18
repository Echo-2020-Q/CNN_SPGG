"""
PlannerNet: 卷积式策略 + 价值网络，负责把环境状态映射为每个格点的分配参数。

可以把它理解成“制度设计者的大脑”：

- 输入 state: tensor shape (B, C=3, L, L)
  - 通道顺序为 [Stra_now, Stra_prev, P_center]，由 env.get_state 构造；
  - Stra_* 为 0/1（合作者=1，背叛者=0）。
- 输出 alpha: (B, 5, L, L)
  - 本实现把策略头输出经过 softplus，得到严格 >0 的参数；
  - 这些参数可视为 Dirichlet 的浓度参数 (alpha)，适用于在动作空间为“概率向量”的场景；
  - Actor 在 worker.py 中用它构建每个格点的 Dirichlet 分布并采样出 5 维概率向量。
- 输出 value: (B,)
  - 对整张棋盘的价值估计 V(s)，由 learner 作为 critic 目标，用于 V-trace / A3C 等算法。

在当前项目中我们采用 Dirichlet 参数化：网络输出的 `alpha` 代表每个格点的 5 维浓度参数，
Actor 用它构建 Dirichlet 分布然后采样得到每格点的分配比例（pi_field）。

数值注意事项：
- 使用 softplus 保证 alpha>0；再加小常数避免极小值导致数值不稳定；
- 当动作是概率向量时，Dirichlet 很自然；但训练时需要注意 log_prob 的定义、
  行为 policy（behavior）与目标 policy 的一致编码，以便重要性比率（V-trace）正确计算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlannerNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        # 主干卷积网络：提取局部与全局特征
        # 对局部博弈问题，卷积网络能捕获邻域交互关系（例如格点四邻居）
        self.body = nn.Sequential(
            # 大感受野的第一层（kernel_size=5）+ padding 保持尺寸
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 两层 3x3 提取更深层的语义特征
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 策略头：1x1 卷积把特征通道base_channels(后文的注释base都是指base_channels)压到 5 个输出通道（对每个格点输出 5 个数）
        # 这 5 个数经过 softplus -> alpha 用作 Dirichlet 的浓度参数
        self.policy_head = nn.Conv2d(base_channels, 5, kernel_size=1)

        # 价值头：先对空间做全局平均池化，再用 MLP 输出单个标量 value
        # 这样 value 是对整张棋盘的评估（全局视角）
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 输入 (B,base,H,W) -> (B,base,1,1)
            nn.Flatten(),              # -> (B, base)
            nn.Linear(base_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        前向计算：
        输入 x: (B, 3, L, L)
        返回 (alpha, value)
          - alpha: (B, 3, L, L), 所有元素 > 0，可作为 Dirichlet 的浓度参数
          - value: (B,)         状态值
        """
        feat = self.body(x)                          # (B, base, L, L)
        raw_alpha = self.policy_head(feat)           # (B,3,L,L), 实数
        # softplus 将实数映射到正数，适合做浓度参数；加小常数避免靠近 0
        alpha = F.softplus(raw_alpha) + 1e-3         # (B,3,L,L)
        value = self.value_head(feat).squeeze(-1)    # (B,)
        return alpha, value
