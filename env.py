import numpy as np


class PublicGoodsEnv:
    """
    空间公共物品博弈环境（局部制度场版）。

    可以粗略理解为：
    - 棋盘: L×L 的格点，每个格点代表一个个体（agent）；
    - 策略: 每个体要么合作者 C=1，要么背叛者 D=0；
    - 小组博弈: 以任意格点 (i,j) 为中心，和它的上下左右共 5 个个体组成“小组”；
    - 公共池: 每个合作者往公共池投入 1，公共池被放大 r 倍，再按 planner 给定的 5 维比例向量在组内分配；
    - 演化: 个体根据邻居的收益差异，通过 Fermi 规则更新自己的策略（向收益高的邻居学习）。

    planner 的角色：
    - 对每个格点 (i,j) 输出一个 5 维向量 π_ij（mid, up, down, left, right）；
    - 这个向量控制以 (i,j) 为中心那一组公共池如何在 5 个个体之间分配；
    - 通过学习 π_field，planner 试图引导系统形成高合作率 / 高总体收益的制度。
    """
    def __init__(self, L=32, r=1.4, R_decay=0.10, use_cumulative_planner_reward=True):
        """
        Args:
            L: 网格边长
            r: 公共物品因子
            R_decay: 每回合累计资源衰减比例（例如 0.10 表示每回合减少 10%）
            use_cumulative_planner_reward: 若为 True，则 planner 奖励按每回合平均净收益累加并返回累计值；
                                           若为 False，则返回每回合的平均净收益（不累加）。
        """
        self.L = L
        self.r = r

        # 当前策略：0=D, 1=C
        self.strategy = np.random.randint(0, 2, size=(L, L), dtype=np.int8)

        # 上一轮策略（初始化时可以先设为当前策略）
        self.prev_strategy = self.strategy.copy()

        # 累计资源
        self.R = np.zeros((L, L), dtype=np.float32)

        # 当轮收益 r_i(t)
        self.r_t = np.zeros((L, L), dtype=np.float32)

        # 每个格点为“中心小组”时的公共池大小 P_ij = r * n_c(ij)
        # （即以 (i,j) 为中心的小组的 P）
        self.P_center = np.zeros((L, L), dtype=np.float32)

        # planner 行为控制
        self.use_cumulative_planner_reward = bool(use_cumulative_planner_reward)
        # planner 累计奖励（按每回合平均净收益累加）
        self.planner_cum_reward = 0.0

        # 每回合累计资源衰减比例
        self.R_decay = float(R_decay)

        # 预先缓存 [0, L) 的索引数组，后续做周期边界映射时无需重复构建
        self.idxs = np.arange(L)
    
    # ==============================================================
    def get_state(self):
        """
        构造 CNN 输入状态（提供给 PlannerNet）：
        1) 当前策略：C_now, D_now（one-hot 两通道）
        2) 上一轮策略：C_prev, D_prev（方便网络感知“策略变化趋势”）
        3) 当前公共池：P_center（以每个格点为中心小组的 P = r * n_c）

        输出 shape: (5, L, L)
        """
        # 当前轮
        C_now = (self.strategy == 1).astype(np.float32)
        D_now = (self.strategy == 0).astype(np.float32)

        # 上一轮
        C_prev = (self.prev_strategy == 1).astype(np.float32)
        D_prev = (self.prev_strategy == 0).astype(np.float32)

        # 当前公共池（以各点为中心的小组的P）
        P_map = self.P_center.astype(np.float32)

        state = np.stack([C_now, D_now, C_prev, D_prev, P_map], axis=0)
        return state

    def _update_strategy_fermi(self, beta=1.0):
        """
        费米更新规则（同步更新版本）：
        每个个体 i 随机选一个邻居 j，根据 r_j - r_i 的差值决定是否模仿 j 的策略。

        直观理解：
        - 若邻居收益更高 (r_j > r_i)，模仿概率接近 1；
        - 若邻居收益更低 (r_j < r_i)，模仿概率接近 0；
        - beta 控制“理性程度”：beta 越大，对收益差越敏感。
        """
        L = self.L
        new_strategy = self.strategy.copy()

        for i in range(L):
            for j in range(L):
                # 1. 随机选一个邻居（周期边界）
                #   邻居集合：上、下、左、右（也可以把自己也算进去）
                neighs = [
                    ((i - 1) % L, j),
                    ((i + 1) % L, j),
                    (i, (j - 1) % L),
                    (i, (j + 1) % L),
                ]
                xj, yj = neighs[np.random.randint(len(neighs))]

                ri = self.r_t[i, j]
                rj = self.r_t[xj, yj]

                # 2. 费米函数给出模仿概率
                diff = rj - ri
                prob = 1.0 / (1.0 + np.exp(-beta * diff))

                # 3. 以 prob 概率采样是否模仿邻居 j
                if np.random.rand() < prob:
                    new_strategy[i, j] = self.strategy[xj, yj]

        self.strategy = new_strategy
    # ==============================================================
    def step(self, pi_field):
        """
        执行一个时间步（planner已输出π_field）。
        π_field: shape (L, L, 5)，每个格点的五维分配比例（mid, up, down, left, right）。

        返回:
            new_state: 下一时刻的状态 (5, L, L)，用于下一步 CNN 输入；
            planner_reward: 本回合 planner 的奖励；
            info: 一些统计量，用于观测系统状态（如合作率等）。
        """
        L, r = self.L, self.r
        new_r = np.zeros_like(self.R)   # 存 r_i(t)

        # 清空上一轮的 P_center
        self.P_center.fill(0.0)

        # --------- 1. 组内公共品博弈，累积 r_i(t) ----------
        # 这里遍历“所有以 (i,j) 为中心的小组”，每个小组都产生一个公共池并在 5 个成员之间分配。
        # 注意：同一个个体会出现在多个小组中，因此 new_r[x,y] 会被多次累加。
        for i in range(L):
            for j in range(L):
                # 当前小组的五个成员坐标（周期边界）
                up    = self.idxs[(i - 1) % L], j
                down  = self.idxs[(i + 1) % L], j
                left  = i, self.idxs[(j - 1) % L]
                right = i, self.idxs[(j + 1) % L]
                mid   = (i, j)
                group_coords = [mid, up, down, left, right]

                # 合作者数量 n_c：统计该小组中策略为 C 的个体数
                n_c = sum(self.strategy[x, y] == 1 for x, y in group_coords)

                # 公共池 P = r * n_c
                P = r * n_c
                
                # 记录以 (i,j) 为中心小组的 P 大小
                self.P_center[i, j] = P

                # 当前小组的局部分配比例向量 π_ij (mid, up, down, left, right)
                # 注意：外部虽然应该已经给出概率向量，但这里仍做一次归一化以防数值偏差。
                pi_vec = pi_field[i, j]          # shape (5,)
                pi_vec = pi_vec / (pi_vec.sum() + 1e-8)  # 保险归一化

                # 结算本小组对每个成员的收益
                for k, (x, y) in enumerate(group_coords):
                    income = pi_vec[k] * P
                    if self.strategy[x, y] == 1:   # 合作者扣成本1
                        income -= 1.0
                    new_r[x, y] += income

        # --------- 2. 更新累计资源 R_i(t+1) = (1 - decay) * R_i(t) + r_i(t) ----------
        self.r_t = new_r
        # 每回合减少当前累计资源的 10%（或 self.R_decay）后再加上本轮收益
        self.R = (1.0 - self.R_decay) * self.R + new_r

        # --------- 3. 计算 planner 奖励 ----------
        # 这里选择“本回合平均净收益”作为奖励信号：
        #   total_P          = 本回合所有小组公共池之和   = r * 总合作者出资次数
        #   5 * total_coop   = 系统中所有合作者的总成本（每个合作者参与 5 个小组）
        #   net_total        = total_P - 5 * total_cooperators
        #   avg_net          = net_total / (L^2)
        # 如果 use_cumulative_planner_reward=True，则对 avg_net 做时间累加，
        # 让 planner 在长期尺度上优化制度；否则就只看单步表现。
        total_P = float(self.P_center.sum())
        total_cooperators = int((self.strategy == 1).sum())
        net_total = total_P - 5.0 * float(total_cooperators)
        avg_net = net_total / float(L * L)

        # 根据开关决定是否累加 planner reward
        if self.use_cumulative_planner_reward:
            # 把每回合的平均净收益累加到累计奖励
            self.planner_cum_reward += avg_net
            planner_reward = float(self.planner_cum_reward)
        else:
            # 只返回本回合的平均净收益（不累加）
            planner_reward = float(avg_net)

        # 同时保留 avg_R_new 供 info 使用
        avg_R_new = float(self.R.mean())
        
        # ====== 策略更新前，先保存当前策略为 prev_strategy ======
        self.prev_strategy = self.strategy.copy()

        # --------- 4. 更新个体策略：Fermi 复制规则 ----------
        self._update_strategy_fermi(beta=1.0)

        info = {
            "avg_R": avg_R_new,
            "avg_r": new_r.mean(),
            "avg_net": avg_net,           # 本回合的平均净收益（total_P - 5*#C）/L^2
            "f_C":  self.strategy.mean(),  # 合作率
        }

        return self.get_state(), planner_reward, info

    def reset(self):
        # 重置策略、资源、r_t、P_center、prev_strategy 等
        self.strategy = np.random.randint(0, 2, size=(self.L, self.L), dtype=np.int8)
        self.prev_strategy = self.strategy.copy()
        self.R.fill(0.0)
        self.r_t.fill(0.0)
        self.P_center.fill(0.0)
        # 重置 planner 累计奖励（如果有使用累计式奖励）
        self.planner_cum_reward = 0.0
        return self.get_state()


