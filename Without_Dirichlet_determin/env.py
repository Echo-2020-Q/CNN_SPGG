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

    本环境版本与 Dirichlet 随机策略版接口保持兼容，
    以便在 TD3 / 确定性 Actor 框架下直接复用。
    额外机制：个体有“累计资源”，若资源不足合作成本则无法合作（即便策略位为 1 也视为 D）。
    """
    def __init__(
        self,
        L=32,
        r=1.4,
        R_decay=0.10,
        use_cumulative_planner_reward=True,
        episode_length=500,
        coop_cost=5.0,
        initial_R=50,
        investment_mode: str = "fixed",
        investment_tau: float = 0.5,
        investment_cmax: float = 5.0,
    ):
        """
        Args:
            L: 网格边长
            r: 公共物品因子
            R_decay: 每回合累计资源衰减比例（例如 0.10 表示每回合减少 10%）
            use_cumulative_planner_reward: 若为 True，则 planner 奖励按每回合平均净收益累加并返回累计值；
                                           若为 False，则返回每回合的平均净收益（不累加）。
            coop_cost: 合作所需成本。若累计资源 R < coop_cost，则本轮无法合作（即使策略位为 1 也视为 D）。
            initial_R: 初始累计资源，避免开局全部无法合作。
        """
        self.L = L
        # r 可能是 numpy/tensor，这里统一转成 Python float 方便后续做标量运算
        # 使用 np.asarray(r).item() 可避免 "only one element tensors" 报错
        self.r = float(np.asarray(r).item())
        self.coop_cost = float(coop_cost)
        self.initial_R = float(initial_R)
        mode = str(investment_mode).strip().lower().replace(" ", "_")
        self.investment_mode = mode
        self.invest_tau = float(investment_tau)
        self.invest_cmax = float(investment_cmax)

        # 当前策略：0=D, 1=C
        self.strategy = np.random.randint(0, 2, size=(L, L), dtype=np.int8)

        # 上一轮策略（初始化时可以先设为当前策略）
        self.prev_strategy = self.strategy.copy()

        # 累计资源
        self.R = np.full((L, L), fill_value=self.initial_R, dtype=np.float32)

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

        # Episode 相关：最大的演化步数 T_max（episode_length），以及当前步计数 t
        # 每次 reset() 会把 self.t 归零，step() 中自增直到达到 episode_length 即 done=True。
        self.episode_length = int(episode_length)
        self.t = 0
    
    # ==============================================================
    def get_state(self):
        """
        构造 CNN 输入状态（提供给 PlannerNet）：
        1) 当前可合作策略：Stra_now（资源足且策略为 C 的为 1，否则 0）
        2) 上一轮策略：Stra_prev（上一轮策略为 C 的为 1，否则 0）
        3) 当前公共池归一化：P_norm
           - fixed: 分母 5*r
           - fixed_add_proportion: 分母 r*(coop_cost + C_max)
        4) 当前累计资源归一化：R_norm，分母为模式对应的稳态上界

        输出 shape: (4, L, L)
        """
        # 当前轮与上一轮的策略（单通道布尔 -> float32），资源不足 coop_cost 视为 D
        # 可合作标记：策略为 C 且资源足够
        can_cooperate = (self.strategy == 1) & (self.R >= self.coop_cost)
        stra_now = can_cooperate.astype(np.float32)
        stra_prev = (self.prev_strategy == 1).astype(np.float32)

        if self.investment_mode == "fixed_add_proportion":
            # P_norm 分母：r * (coop_cost + C_max)，对应单个小组的理论最大公共池
            P_norm = self.P_center.astype(np.float32) / (float(self.r) * (self.coop_cost + self.invest_cmax) + 1e-8)
            # 稳态上界估计：(1/(1-R_decay)) * (r-1) * (coop_cost + C_max)
            steady_max_R = (float(self.r) - 1.0) * (self.coop_cost + self.invest_cmax) / max(1e-8, 1.0 - self.R_decay)
        else:
            P_norm = self.P_center.astype(np.float32) / (5.0 * float(self.r) + 1e-8)
            steady_max_R = 5.0 * max(float(self.r) - 1.0, 0.0) / (self.R_decay + 1e-8)

        scale_R = max(steady_max_R, self.coop_cost * 10.0)
        R_norm = self.R.astype(np.float32) / (scale_R + 1e-8)

        state = np.stack([stra_now, stra_prev, P_norm, R_norm], axis=0)
        return state

    def _update_strategy_fermi(self, beta=1.0):
        """
        费米更新规则（同步更新版本，向量化）：
        每个个体随机选一个邻居，根据 r_j - r_i 的差值决定是否模仿邻居的策略。
        """
        L = self.L
        # 随机方向：0=up,1=down,2=left,3=right
        dir_idx = np.random.randint(0, 4, size=(L, L))

        # 预先 roll 出四个方向的 r_t 和 strategy
        r_up = np.roll(self.r_t, shift=1, axis=0)
        r_down = np.roll(self.r_t, shift=-1, axis=0)
        r_left = np.roll(self.r_t, shift=1, axis=1)
        r_right = np.roll(self.r_t, shift=-1, axis=1)

        s_up = np.roll(self.strategy, shift=1, axis=0)
        s_down = np.roll(self.strategy, shift=-1, axis=0)
        s_left = np.roll(self.strategy, shift=1, axis=1)
        s_right = np.roll(self.strategy, shift=-1, axis=1)

        # 按随机方向选邻居的 r_j / s_j
        neigh_r = np.take_along_axis(
            np.stack([r_up, r_down, r_left, r_right], axis=-1), dir_idx[..., None], axis=-1
        ).squeeze(-1)
        neigh_s = np.take_along_axis(
            np.stack([s_up, s_down, s_left, s_right], axis=-1), dir_idx[..., None], axis=-1
        ).squeeze(-1)

        ri = self.r_t
        diff = neigh_r - ri
        x = -beta * diff
        x = np.clip(x, -60.0, 60.0)
        prob = 1.0 / (1.0 + np.exp(x))

        # 采样是否模仿邻居
        mask = np.random.rand(L, L) < prob
        new_strategy = self.strategy.copy()
        new_strategy[mask] = neigh_s[mask]

        # 资源不足 coop_cost 的个体无法保持 / 切换到合作，强制视为 D
        new_strategy[self.R < self.coop_cost] = 0
        self.strategy = new_strategy
    # ==============================================================
    def reset(self):
        """单环境重置。"""
        self.strategy = np.random.randint(0, 2, size=(self.L, self.L), dtype=np.int8)
        self.prev_strategy = self.strategy.copy()
        self.R.fill(self.initial_R)
        self.r_t.fill(0.0)
        self.P_center.fill(0.0)
        self.planner_cum_reward = 0.0
        self.t = 0
        return self.get_state()
    # ==============================================================
    def step(self, pi_field):
        """
        执行一个时间步（planner已输出π_field）。
        π_field: shape (L, L, 5)，每个格点的五维分配比例（mid, up, down, left, right）。

        返回:
            new_state: 下一时刻的状态 (4, L, L)，用于下一步 CNN 输入；
            planner_reward: 本回合 planner 的奖励；
            done: 是否到达本 episode 结束（达到最大演化步数 T_max）；
            info: 一些统计量，用于观测系统状态（如合作率等）。
        """
        L, r = self.L, self.r
        new_r = np.zeros_like(self.R, dtype=np.float32)   # 存 r_i(t)

        # 清空上一轮的 P_center
        self.P_center.fill(0.0)

        # --------- 1. 组内公共品博弈，累积 r_i(t)（向量化实现） ----------
        # can_cooperate: 当前轮真正能合作的个体（策略为 C 且资源 >= coop_cost）
        if self.investment_mode == "fixed_add_proportion":
            steady_max_R = (float(self.r) - 1.0) * (self.coop_cost + self.invest_cmax) / max(1e-8, 1.0 - self.R_decay)
        else:
            steady_max_R = 5.0 * max(float(self.r) - 1.0, 0.0) / (self.R_decay + 1e-8)
        R_cap = max(steady_max_R * 20.0, self.coop_cost * 20.0)
        self.R = np.clip(self.R, 0.0, R_cap)
        can_cooperate = (self.strategy == 1) & (self.R >= self.coop_cost)

        # 按模式计算个体投入
        if self.investment_mode == "fixed_add_proportion":
            # R_invest = coop_cost + C_max * tanh(max(0, τ * (R - coop_cost) / C_max))
            invest_extra = np.maximum(0.0, self.R - self.coop_cost)
            invest_agent = self.coop_cost + self.invest_cmax * np.tanh((self.invest_tau * invest_extra) / max(1e-8, self.invest_cmax))
            invest_agent = np.where(can_cooperate, invest_agent, 0.0).astype(np.float32)
        else:
            invest_agent = can_cooperate.astype(np.float32)  # fixed: 原逻辑，每次参与投入 1

        # 以每个格点为“中心小组”视角，计算该小组中 5 个成员的可合作标记
        mid_inv = invest_agent.astype(np.float32)                     # (L,L)
        up_inv = np.roll(invest_agent, shift=1, axis=0).astype(np.float32)    # 上邻居位于 (i-1,j)
        down_inv = np.roll(invest_agent, shift=-1, axis=0).astype(np.float32) # 下邻居位于 (i+1,j)
        left_inv = np.roll(invest_agent, shift=1, axis=1).astype(np.float32)  # 左邻居位于 (i,j-1)
        right_inv = np.roll(invest_agent, shift=-1, axis=1).astype(np.float32)# 右邻居位于 (i,j+1)

        # 公共池 P = r * (各角色投入之和)，记录到以 (i,j) 为中心的小组公共池 self.P_center
        n_invest = mid_inv + up_inv + down_inv + left_inv + right_inv      # (L,L)
        self.P_center[:, :] = r * n_invest.astype(np.float32)

        # 当前小组的局部分配比例向量 π_ij (mid, up, down, left, right)
        # 注意：外部虽然应该已经给出概率向量，但这里仍做一次归一化以防数值偏差。
        pi = np.asarray(pi_field, dtype=np.float32)  # (L,L,5)
        denom = pi.sum(axis=-1, keepdims=True) + 1e-8
        pi = pi / denom

        # 每个小组对 5 个成员的收益（尚未扣成本）：income_role = pi_role * P_center
        P = self.P_center.astype(np.float32)
        income_mid_center = P * pi[:, :, 0]
        income_up_center = P * pi[:, :, 1]
        income_down_center = P * pi[:, :, 2]
        income_left_center = P * pi[:, :, 3]
        income_right_center = P * pi[:, :, 4]

        # 将“以小组为中心”的收益映射到具体个体坐标：
        # - 自己是中心：不需要平移
        # - 自己是某个邻居：把对应 role 的地图 roll 回该邻居坐标
        income_mid_agent = income_mid_center
        income_up_agent = np.roll(income_up_center, shift=-1, axis=0)     # 每个个体作为某组的 up 成员
        income_down_agent = np.roll(income_down_center, shift=1, axis=0)  # 作为 down 成员
        income_left_agent = np.roll(income_left_center, shift=-1, axis=1) # 作为 left 成员
        income_right_agent = np.roll(income_right_center, shift=1, axis=1)# 作为 right 成员

        income_total = (
            income_mid_agent
            + income_up_agent
            + income_down_agent
            + income_left_agent
            + income_right_agent
        )

        # 成本：按投入额扣减（与分配映射同样的 roll 逻辑）
        cost_mid_center = mid_inv
        cost_up_center = up_inv
        cost_down_center = down_inv
        cost_left_center = left_inv
        cost_right_center = right_inv

        cost_mid_agent = cost_mid_center
        cost_up_agent = np.roll(cost_up_center, shift=-1, axis=0)
        cost_down_agent = np.roll(cost_down_center, shift=1, axis=0)
        cost_left_agent = np.roll(cost_left_center, shift=-1, axis=1)
        cost_right_agent = np.roll(cost_right_center, shift=1, axis=1)

        cost_total = (
            cost_mid_agent
            + cost_up_agent
            + cost_down_agent
            + cost_left_agent
            + cost_right_agent
        )

        new_r[:, :] = income_total - cost_total

        # --------- 2. 更新累计资源 R_i(t+1) = (1 - decay) * R_i(t) + r_i(t) ----------
        self.r_t = new_r
        # 每回合减少当前累计资源的 10%（或 self.R_decay）后再加上本轮收益
        self.R = (1.0 - self.R_decay) * self.R + new_r
        self.R = np.clip(self.R, 0.0, R_cap)

        # --------- 3. 计算 planner 奖励 ----------
        # 这里选择“本回合平均净收益”作为奖励信号：
        # 如果 use_cumulative_planner_reward=True，则对 avg_net 做时间累加；
        # 否则只看单步表现。归一化分母随投入模式变化：
        # - fixed:     scale=5*(r-1)
        # - add_prop:  scale=(r-1)*(coop_cost+C_max)
        total_P = float(self.P_center.sum())
        total_invest = float(cost_total.sum())
        net_total = total_P - total_invest
        avg_net = net_total / float(L * L)

        if self.investment_mode == "fixed_add_proportion":
            scale = max((self.r - 1.0) * (self.coop_cost + self.invest_cmax), 1e-8)
        else:
            scale = 5.0 * max(self.r - 1.0, 1e-8)
        norm_avg_net = avg_net / scale

        # 根据开关决定是否累加 planner reward（在 TD3 设置中通常关闭累加）但是这是是默认开启累加的
        if self.use_cumulative_planner_reward:
            # 把“归一化后的平均净收益”累加到累计奖励
            self.planner_cum_reward += norm_avg_net
            planner_reward = float(self.planner_cum_reward)
        else:
            # 只返回本回合的归一化平均净收益（不累加）
            planner_reward = float(norm_avg_net)

        # 同时保留 avg_R_new 供 info 使用
        avg_R_new = float(self.R.mean())
        
        # ====== 策略更新前，先保存“上一轮的可合作状态”作为 prev_strategy ======
        self.prev_strategy = ((self.strategy == 1) & (self.R >= self.coop_cost)).astype(np.int8)

        # --------- 4. 更新个体策略：Fermi 复制规则 ----------
        self._update_strategy_fermi(beta=1.0)

        # Episode 步数推进，并判断是否终止（达到最大演化步数）
        self.t += 1
        done = self.t >= self.episode_length

        info = {
            "avg_R": avg_R_new,
            "avg_r": new_r.mean(),
            "avg_net": avg_net,           # 本回合的平均净收益（total_P - 5*#C）/L^2
            "f_C":  ((self.strategy == 1) & (self.R >= self.coop_cost)).mean(),  # 实际可合作率
            "t": self.t,
            "done": done,
        }

        return self.get_state(), planner_reward, done, info


class BatchedPublicGoodsEnv:
    """
    单进程批量环境：在同一个进程内同时推进 batch_size 个棋盘，避免多进程 IPC 开销。
    所有 env 共享相同的 episode_length，时间步同步推进。
    """

    def __init__(
        self,
        batch_size: int,
        L=32,
        r=4,
        R_decay=0.10,
        use_cumulative_planner_reward=False,
        episode_length=150,
        coop_cost=5.0,
        initial_R=50,
        investment_mode: str = "fixed",
        investment_tau: float = 0.5,
        investment_cmax: float = 5.0,
    ):
        self.batch_size = int(batch_size)
        self.L = L
        self.r = float(np.asarray(r).item())
        self.coop_cost = float(coop_cost)
        self.initial_R = float(initial_R)
        self.R_decay = float(R_decay)
        self.use_cumulative_planner_reward = bool(use_cumulative_planner_reward)
        self.episode_length = int(episode_length)
        mode = str(investment_mode).strip().lower().replace(" ", "_")
        self.investment_mode = mode
        self.invest_tau = float(investment_tau)
        self.invest_cmax = float(investment_cmax)

        self.strategy = np.random.randint(0, 2, size=(self.batch_size, L, L), dtype=np.int8)
        self.prev_strategy = self.strategy.copy()
        self.R = np.full((self.batch_size, L, L), fill_value=self.initial_R, dtype=np.float32)
        self.r_t = np.zeros((self.batch_size, L, L), dtype=np.float32)
        self.P_center = np.zeros((self.batch_size, L, L), dtype=np.float32)
        self.planner_cum_reward = np.zeros((self.batch_size,), dtype=np.float32)
        self.t = 0

    def get_state(self):
        can_cooperate = (self.strategy == 1) & (self.R >= self.coop_cost)
        stra_now = can_cooperate.astype(np.float32)
        stra_prev = (self.prev_strategy == 1).astype(np.float32)
        if self.investment_mode == "fixed_add_proportion":
            P_norm = self.P_center.astype(np.float32) / (float(self.r) * (self.coop_cost + self.invest_cmax) + 1e-8)
            steady_max_R = (float(self.r) - 1.0) * (self.coop_cost + self.invest_cmax) / max(1e-8, 1.0 - self.R_decay)
        else:
            P_norm = self.P_center.astype(np.float32) / (5.0 * float(self.r) + 1e-8)
            steady_max_R = 5.0 * max(float(self.r) - 1.0, 0.0) / (self.R_decay + 1e-8)
        scale_R = max(steady_max_R, self.coop_cost * 10.0)
        R_norm = self.R.astype(np.float32) / (scale_R + 1e-8)
        state = np.stack([stra_now, stra_prev, P_norm, R_norm], axis=1)  # (B,4,L,L)
        return state

    def step(self, pi_field):
        """
        pi_field: (B, L, L, 5)
        返回: state_next (B,4,L,L), reward (B,), done(bool), info(dict汇总均值)
        """
        B, L = self.batch_size, self.L
        new_r = np.zeros_like(self.R, dtype=np.float32)
        self.P_center.fill(0.0)

        if self.investment_mode == "fixed_add_proportion":
            steady_max_R = (float(self.r) - 1.0) * (self.coop_cost + self.invest_cmax) / max(1e-8, 1.0 - self.R_decay)
        else:
            steady_max_R = 5.0 * max(float(self.r) - 1.0, 0.0) / (self.R_decay + 1e-8)
        R_cap = max(steady_max_R * 20.0, self.coop_cost * 20.0)
        self.R = np.clip(self.R, 0.0, R_cap)

        can_cooperate = (self.strategy == 1) & (self.R >= self.coop_cost)
        if self.investment_mode == "fixed_add_proportion":
            invest_extra = np.maximum(0.0, self.R - self.coop_cost)
            invest_agent = self.coop_cost + self.invest_cmax * np.tanh((self.invest_tau * invest_extra) / max(1e-8, self.invest_cmax))
            invest_agent = np.where(can_cooperate, invest_agent, 0.0).astype(np.float32)
        else:
            invest_agent = can_cooperate.astype(np.float32)

        mid_inv = invest_agent.astype(np.float32)
        up_inv = np.roll(invest_agent, shift=1, axis=1).astype(np.float32)
        down_inv = np.roll(invest_agent, shift=-1, axis=1).astype(np.float32)
        left_inv = np.roll(invest_agent, shift=1, axis=2).astype(np.float32)
        right_inv = np.roll(invest_agent, shift=-1, axis=2).astype(np.float32)

        n_invest = mid_inv + up_inv + down_inv + left_inv + right_inv
        self.P_center[:, :, :] = self.r * n_invest

        # 归一化分配比例，防止数值漂移
        pi = np.asarray(pi_field, dtype=np.float32)
        denom = pi.sum(axis=-1, keepdims=True) + 1e-8
        pi = pi / denom

        P = self.P_center.astype(np.float32)
        income_mid_center = P * pi[:, :, :, 0]
        income_up_center = P * pi[:, :, :, 1]
        income_down_center = P * pi[:, :, :, 2]
        income_left_center = P * pi[:, :, :, 3]
        income_right_center = P * pi[:, :, :, 4]

        income_mid_agent = income_mid_center
        income_up_agent = np.roll(income_up_center, shift=-1, axis=1)
        income_down_agent = np.roll(income_down_center, shift=1, axis=1)
        income_left_agent = np.roll(income_left_center, shift=-1, axis=2)
        income_right_agent = np.roll(income_right_center, shift=1, axis=2)

        income_total = (
            income_mid_agent
            + income_up_agent
            + income_down_agent
            + income_left_agent
            + income_right_agent
        )

        cost_mid_agent = mid_inv
        cost_up_agent = np.roll(up_inv, shift=-1, axis=1)
        cost_down_agent = np.roll(down_inv, shift=1, axis=1)
        cost_left_agent = np.roll(left_inv, shift=-1, axis=2)
        cost_right_agent = np.roll(right_inv, shift=1, axis=2)

        cost_total = (
            cost_mid_agent
            + cost_up_agent
            + cost_down_agent
            + cost_left_agent
            + cost_right_agent
        )

        # 每个个体的净收益 = 收入 - 成本
        new_r[:, :, :] = income_total - cost_total

        self.r_t = new_r
        self.R = (1.0 - self.R_decay) * self.R + new_r
        self.R = np.clip(self.R, 0.0, R_cap)

        total_P = self.P_center.sum(axis=(1, 2)).astype(np.float32)
        total_invest = cost_total.sum(axis=(1, 2)).astype(np.float32)
        net_total = total_P - total_invest
        avg_net = net_total / float(L * L)
        if self.investment_mode == "fixed_add_proportion":
            scale = max((self.r - 1.0) * (self.coop_cost + self.invest_cmax), 1e-8)
        else:
            scale = 5.0 * max(self.r - 1.0, 1e-8)
        norm_avg_net = avg_net / scale

        if self.use_cumulative_planner_reward:
            self.planner_cum_reward += norm_avg_net
            planner_reward = self.planner_cum_reward.copy()
        else:
            planner_reward = norm_avg_net

        avg_R_new = self.R.mean()
        self.prev_strategy = ((self.strategy == 1) & (self.R >= self.coop_cost)).astype(np.int8)

        # --------- Fermi 更新（批量向量化） ---------
        # 随机方向：0=up,1=down,2=left,3=right
        dir_idx = np.random.randint(0, 4, size=(B, L, L))
        r_up = np.roll(self.r_t, shift=1, axis=1)
        r_down = np.roll(self.r_t, shift=-1, axis=1)
        r_left = np.roll(self.r_t, shift=1, axis=2)
        r_right = np.roll(self.r_t, shift=-1, axis=2)

        s_up = np.roll(self.strategy, shift=1, axis=1)
        s_down = np.roll(self.strategy, shift=-1, axis=1)
        s_left = np.roll(self.strategy, shift=1, axis=2)
        s_right = np.roll(self.strategy, shift=-1, axis=2)

        neigh_r = np.take_along_axis(
            np.stack([r_up, r_down, r_left, r_right], axis=-1), dir_idx[..., None], axis=-1
        ).squeeze(-1)
        neigh_s = np.take_along_axis(
            np.stack([s_up, s_down, s_left, s_right], axis=-1), dir_idx[..., None], axis=-1
        ).squeeze(-1)

        diff = neigh_r - self.r_t
        x = -1.0 * diff
        x = np.clip(x, -60.0, 60.0)
        prob = 1.0 / (1.0 + np.exp(x))

        mask = np.random.rand(B, L, L) < prob
        new_strategy = self.strategy.copy()
        new_strategy[mask] = neigh_s[mask]
        new_strategy[self.R < self.coop_cost] = 0
        self.strategy = new_strategy

        self.t += 1
        done = self.t >= self.episode_length

        info = {
            "avg_R": float(avg_R_new),
            "avg_r": float(new_r.mean()),
            "avg_net": float(avg_net.mean()),
            "f_C": float(((self.strategy == 1) & (self.R >= self.coop_cost)).mean()),
            "t": self.t,
            "done": done,
        }

        return self.get_state(), planner_reward, done, info

    def reset(self):
        """批量重置：重新随机策略/资源，清零计数，返回初始状态。"""
        self.strategy = np.random.randint(0, 2, size=(self.batch_size, self.L, self.L), dtype=np.int8)
        self.prev_strategy = self.strategy.copy()
        self.R.fill(self.initial_R)
        self.r_t.fill(0.0)
        self.P_center.fill(0.0)
        self.planner_cum_reward.fill(0.0)
        self.t = 0
        return self.get_state()
