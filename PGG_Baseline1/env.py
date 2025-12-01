import datetime
from typing import Optional

import numpy as np
import datetime


class PublicGoodsEnv:
    """
    空间公共物品博弈环境（局部制度场版），与 Without_Dirichlet_determin/env.py 保持一致。

    核心规则：
    - 棋盘: L×L 个体，策略 C=1 / D=0；
    - 组：以任一点 (i,j) 为中心的小组包含 5 人（自己+上下左右）；
    - 公共池：P = r * n_c，其中 n_c 只统计“策略为 C 且资源充足”的个体；
    - 分配：planner 提供 π_field，控制以 (i,j) 为中心的小组公共池如何分给组内 5 人；
    - 成本：只有资源 >= coop_cost 的合作者才支付成本 1，否则视为 D；
    - 资源演化：R_{t+1} = (1 - R_decay) * R_t + r_t，初始为 initial_R。
    """

    def __init__(
        self,
        L=32,
        r=1.4,
        R_decay=0.10,
        use_cumulative_planner_reward=True,
        episode_length=500,
        coop_cost=5.0,
        initial_R=30,
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
        self.r = float(np.asarray(r).item())
        self.coop_cost = float(coop_cost)
        self.initial_R = float(initial_R)

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
        1) stra_now：当前可合作策略（资源足且策略为 C 才算 1）
        2) stra_prev：上一轮可合作策略
        3) P_center_norm：以各点为中心小组的公共池 P，按 5*r 归一化

        输出 shape: (3, L, L)
        """
        # 当前轮与上一轮的策略（单通道布尔 -> float32），资源不足 coop_cost 视为 D
        can_cooperate = (self.strategy == 1) & (self.R >= self.coop_cost)
        stra_now = can_cooperate.astype(np.float32)
        stra_prev = (self.prev_strategy == 1).astype(np.float32)

        # 当前公共池（以各点为中心的小组的P），按理论最大值 5*r 做归一化
        P_map = self.P_center.astype(np.float32) / (5.0 * float(self.r) + 1e-8)

        state = np.stack([stra_now, stra_prev, P_map], axis=0)
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

        # 资源不足 coop_cost 的个体无法保持 / 切换到合作，强制视为 D
        new_strategy[self.R < self.coop_cost] = 0
        self.strategy = new_strategy

    # ==============================================================
    def step(self, pi_field):
        """
        执行一个时间步（planner已输出π_field）。
        π_field: shape (L, L, 5)，每个格点的五维分配比例（mid, up, down, left, right）。

        返回:
            new_state: 下一时刻的状态 (3, L, L)，用于下一步 CNN 输入；
            planner_reward: 本回合 planner 的奖励；
            done: 是否到达本 episode 结束（达到最大演化步数 T_max）；
            info: 一些统计量，用于观测系统状态（如合作率等）。
        """
        L, r = self.L, self.r
        new_r = np.zeros_like(self.R)   # 存 r_i(t)

        # 清空上一轮的 P_center
        self.P_center.fill(0.0)

        # --------- 1. 组内公共品博弈，累积 r_i(t) ----------
        # 遍历所有以 (i,j) 为中心的小组
        for i in range(L):
            for j in range(L):
                # 当前小组的五个成员坐标（周期边界）
                up    = self.idxs[(i - 1) % L], j
                down  = self.idxs[(i + 1) % L], j
                left  = i, self.idxs[(j - 1) % L]
                right = i, self.idxs[(j + 1) % L]
                mid   = (i, j)
                group_coords = [mid, up, down, left, right]

                # 合作者数量 n_c：策略为 C 且资源充足的个体才算合作者
                n_c = sum((self.strategy[x, y] == 1) and (self.R[x, y] >= self.coop_cost) for x, y in group_coords)

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
                    # 只有资源充足且策略为 C 的个体才真正付出成本
                    if (self.strategy[x, y] == 1) and (self.R[x, y] >= self.coop_cost):
                        income -= 1.0
                    new_r[x, y] += income

        # --------- 2. 更新累计资源 R_i(t+1) = (1 - decay) * R_i(t) + r_i(t) ----------
        self.r_t = new_r
        # 每回合减少当前累计资源的 R_decay 后再加上本轮收益
        self.R = (1.0 - self.R_decay) * self.R + new_r

        # --------- 3. 计算 planner 奖励 ----------
        # 这里选择“本回合平均净收益”作为奖励信号：
        #   total_P          = 本回合所有小组公共池之和   = r * 总合作者出资次数
        #   5 * total_coop   = 系统中所有合作者的总成本（每个合作者参与 5 个小组）
        #   net_total        = total_P - 5 * total_cooperators
        #   avg_net          = net_total / (L^2)
        # 如果 use_cumulative_planner_reward=True，则对 avg_net 做时间累加，
        # 否则只看单步表现。
        total_P = float(self.P_center.sum())
        total_cooperators = int(((self.strategy == 1) & (self.R >= self.coop_cost)).sum())
        net_total = total_P - 5.0 * float(total_cooperators)
        avg_net = net_total / float(L * L)

        # 进一步按理论最大值 5*(r-1) 做归一化，得到 per-step 归一化奖励
        scale = 5.0 * max(self.r - 1.0, 1e-8)
        norm_avg_net = avg_net / scale

        if self.use_cumulative_planner_reward:
            self.planner_cum_reward += norm_avg_net
            planner_reward = float(self.planner_cum_reward)
        else:
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

    def reset(self):
        # 重置策略、资源、r_t、P_center、prev_strategy 等
        self.strategy = np.random.randint(0, 2, size=(self.L, self.L), dtype=np.int8)
        self.prev_strategy = self.strategy.copy()
        self.R.fill(self.initial_R)
        self.r_t.fill(0.0)
        self.P_center.fill(0.0)
        # 重置 planner 累计奖励（如果有使用累计式奖励）
        self.planner_cum_reward = 0.0
        # 重置 episode 步数计数
        self.t = 0
        return self.get_state()


# ==============================================================
# 基线策略模拟：平均分配 / 按贡献分配
# ==============================================================

def _build_pi_equal(L: int) -> np.ndarray:
    """
    生成“完全平均分配”的 π_field：
    - 对每个以 (i,j) 为中心的小组，P 在 mid/up/down/left/right 五个方向平均 1/5 分配。
    """
    return np.ones((L, L, 5), dtype=np.float32) / 5.0


def _build_pi_contrib(env: PublicGoodsEnv) -> np.ndarray:
    """
    生成“按贡献分配”的 π_field：
    - 仍然以 (i,j) 为中心形成 5 人小组；
    - 若该组中有 n_c 个“可合作个体”（策略为 C 且 R>=coop_cost）：
        - 按贡献平均分配：这 n_c 个个体各分 P/n_c，对应 π_vec[k] = 1/n_c；
        - 其余（背叛或资源不足者）分到 0；
    - 若 n_c == 0，则退化为平均分配（π_vec = 1/5），避免除零。
    """
    L = env.L
    pi_field = np.zeros((L, L, 5), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            up    = env.idxs[(i - 1) % L], j
            down  = env.idxs[(i + 1) % L], j
            left  = i, env.idxs[(j - 1) % L]
            right = i, env.idxs[(j + 1) % L]
            mid   = (i, j)
            group_coords = [mid, up, down, left, right]

            can_coop_flags = [
                (env.strategy[x, y] == 1) and (env.R[x, y] >= env.coop_cost)
                for (x, y) in group_coords
            ]
            n_c = sum(can_coop_flags)

            if n_c <= 0:
                pi_vec = np.ones(5, dtype=np.float32) / 5.0
            else:
                pi_vec = np.zeros(5, dtype=np.float32)
                share = 1.0 / float(n_c)
                for k, can_c in enumerate(can_coop_flags):
                    if can_c:
                        pi_vec[k] = share

            pi_field[i, j] = pi_vec
    return pi_field


def simulate_baseline(
    L: int = 32,
    r: float = 1.4,
    episode_length: int = 500,
    R_decay: float = 0.10,
    coop_cost: float = 5.0,
    initial_R: float = 30.0,
    alloc_mode: str = "equal",  # "equal" 或 "contrib"
    T: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    在本环境上跑一条基线轨迹（无学习，只是固定分配规则）：
    - alloc_mode="equal":  每个小组公共池在 5 人之间平均分配；
    - alloc_mode="contrib": 只在可合作个体之间平均分配，背叛者和资源不足者得 0。

    记录：
    - f_C(t): 实际可合作率；
    - f_D(t): 1 - f_C(t)；
    - avg_R(t): 平均累计资源；

    返回:
        hist: dict，键包括 "f_C", "f_D", "avg_R"。
    """
    if seed is not None:
        np.random.seed(seed)

    if T is None:
        T = episode_length

    env = PublicGoodsEnv(
        L=L,
        r=r,
        R_decay=R_decay,
        use_cumulative_planner_reward=False,
        episode_length=episode_length,
        coop_cost=coop_cost,
        initial_R=initial_R,
    )
    env.reset()

    hist = {"f_C": [], "f_D": [], "avg_R": []}

    for t in range(1, T + 1):
        if alloc_mode == "equal":
            pi_field = _build_pi_equal(L)
        elif alloc_mode == "contrib":
            pi_field = _build_pi_contrib(env)
        else:
            raise ValueError(f"Unknown alloc_mode: {alloc_mode}")

        _, _, done, info = env.step(pi_field)

        fC = float(info.get("f_C", 0.0))
        fD = 1.0 - fC
        avg_R = float(info.get("avg_R", 0.0))

        hist["f_C"].append(fC)
        hist["f_D"].append(fD)
        hist["avg_R"].append(avg_R)

        if done:
            break

    return hist


def repeat_simulation(
    num_repeat: int,
    **kwargs,
):
    """
    多次独立重复运行 simulate_baseline，返回逐点平均与标准差。

    Args:
        num_repeat: 重复次数。
        kwargs: 传给 simulate_baseline 的其他参数（如 L, r, alloc_mode 等）。

    返回:
        mean_hist, std_hist: 同结构 dict，键包括 "f_C", "f_D", "avg_R"。
    """
    all_results = []
    base_seed = kwargs.pop("seed", 42)
    for i in range(num_repeat):
        hist = simulate_baseline(seed=base_seed + i * 100, **kwargs)
        all_results.append(hist)

    keys = all_results[0].keys()
    mean_hist, std_hist = {}, {}
    for k in keys:
        arr = np.array([h[k] for h in all_results], dtype=np.float32)
        mean_hist[k] = arr.mean(axis=0)
        std_hist[k] = arr.std(axis=0)
    return mean_hist, std_hist


# ==============================================================
# 可视化：f_C/f_D/avg_R 的均值 + 置信区间
# ==============================================================
def plot_baseline_results(
    mean_hist: dict,
    std_hist: dict,
    L: int,
    r: float,
    alloc_mode: str,
    episode_length: int,
    num_repeat: int,
    out_dir: str,
    show_fig: bool = False,
):
    """
    绘制两张图：
    1) f_C, f_D vs t（均值 ±1 标准差的带状区域）
    2) avg_R vs t（均值 ±1 标准差）
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    def _arr(name):
        return np.asarray(mean_hist[name], dtype=float).reshape(-1)

    def _arr_std(name):
        return np.asarray(std_hist[name], dtype=float).reshape(-1)

    fC = _arr("f_C")
    sC = _arr_std("f_C")
    fD = _arr("f_D")
    sD = _arr_std("f_D")
    aR = _arr("avg_R")
    sR = _arr_std("avg_R")

    T_plot = min(len(fC), len(fD), len(aR))
    t = np.arange(T_plot, dtype=float)
    fC, sC = fC[:T_plot], sC[:T_plot]
    fD, sD = fD[:T_plot], sD[:T_plot]
    aR, sR = aR[:T_plot], sR[:T_plot]

    if T_plot == 0:
        # 没有数据可画，直接跳过，避免 matplotlib 在 where 标量时出错
        return

    # 图 1：f_C & f_D
    plt.figure(figsize=(7, 4))
    where_mask = np.ones_like(t, dtype=bool)
    plt.fill_between(
        t,
        np.clip(fC - sC, 0.0, 1.0),
        np.clip(fC + sC, 0.0, 1.0),
        where=where_mask,
                     alpha=0.2, color="tab:blue", label="f_C ± std")
    plt.plot(t, fC, color="tab:blue", linewidth=2, label="f_C (mean)")

    plt.fill_between(
        t,
        np.clip(fD - sD, 0.0, 1.0),
        np.clip(fD + sD, 0.0, 1.0),
        where=where_mask,
                     alpha=0.2, color="tab:orange", label="f_D ± std")
    plt.plot(t, fD, color="tab:orange", linewidth=2, label="f_D (mean)")

    plt.xlabel("Time step")
    plt.ylabel("Fraction")
    plt.title(f"Baseline {alloc_mode}: f_C & f_D (L={L}, r={r}, T={episode_length}, rep={num_repeat})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_fcfd = os.path.join(out_dir, f"baseline_fc_fd_L{L}_r{r:.2f}_{alloc_mode}_T{episode_length}_rep{num_repeat}.png")
    plt.savefig(fname_fcfd, dpi=150)
    if show_fig:
        plt.show()
    plt.close()

    # 图 2：avg_R
    plt.figure(figsize=(7, 4))
    plt.fill_between(
        t, aR - sR, aR + sR, where=where_mask, alpha=0.2, color="tab:green", label="avg_R ± std")
    plt.plot(t, aR, color="tab:green", linewidth=2, label="avg_R (mean)")

    plt.xlabel("Time step")
    plt.ylabel("Average resource R")
    plt.title(f"Baseline {alloc_mode}: avg_R (L={L}, r={r}, T={episode_length}, rep={num_repeat})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_R = os.path.join(out_dir, f"baseline_avgR_L{L}_r{r:.2f}_{alloc_mode}_T{episode_length}_rep{num_repeat}.png")
    plt.savefig(fname_R, dpi=150)
    if show_fig:
        plt.show()
    plt.close()


def run_baseline_grid(
    L_list,
    r_list,
    alloc_modes=("equal", "contrib"),
    num_repeat: int = 20,
    episode_length: int = 150,
    out_dir: str = "results",
):
    """
    扫描多个 (L, r) 以及分配模式，跑基线并画图，同时保存数值结果：
    - 对每个 (L, r, alloc_mode) 组合：
        * 运行 repeat_simulation 得到逐步的 mean/std；
        * 画 f_C/f_D/avg_R 三条曲线（均值 ±1 std）；
        * 计算最后 15% 回合上的 tail 平均：tail_mean_fC, tail_mean_fD, tail_mean_avg_R；
        * 追加到 summary CSV。

    用法示例（在 PGG_Baseline1 目录下）:
        from env import run_baseline_grid
        run_baseline_grid(
            L_list=[25, 30],
            r_list=[2.0, 3.0, 4.0],
            alloc_modes=("equal", "contrib"),
            num_repeat=20,
            episode_length=150,
        )
    """
    import os
    import csv
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)

    # 为本次运行创建时间戳子目录，避免覆盖：results/<timestamp>_<params>/
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    param_tag = f"L{min(L_list)}-{max(L_list)}_r{min(r_list):.2f}-{max(r_list):.2f}_rep{num_repeat}"
    run_dir = os.path.join(out_dir, f"{run_tag}_{param_tag}")
    os.makedirs(run_dir, exist_ok=True)

    summary_path = os.path.join(run_dir, "baseline_summary_Lr_modes.csv")
    # 覆盖写入一份新的汇总 CSV
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "L",
            "r",
            "alloc_mode",
            "episode_length",
            "num_repeat",
            "tail_ratio",
            "tail_steps",
            "tail_mean_fC",
            "tail_mean_fD",
            "tail_mean_avg_R",
        ])

        tail_ratio = 0.15  # 取最后 15% 回合

        for L in L_list:
            for r in r_list:
                for mode in alloc_modes:
                    mean_hist, std_hist = repeat_simulation(
                        num_repeat=num_repeat,
                        L=L,
                        r=r,
                        episode_length=episode_length,
                        alloc_mode=mode,
                    )
                    plot_baseline_results(
                        mean_hist=mean_hist,
                        std_hist=std_hist,
                        L=L,
                        r=r,
                        alloc_mode=mode,
                        episode_length=episode_length,
                    num_repeat=num_repeat,
                    out_dir=run_dir,
                    show_fig=False,
                )

                    # 计算最后 15% 回合上的评价指标
                    fC_arr = np.asarray(mean_hist["f_C"], dtype=float).reshape(-1)
                    fD_arr = np.asarray(mean_hist["f_D"], dtype=float).reshape(-1)
                    R_arr = np.asarray(mean_hist["avg_R"], dtype=float).reshape(-1)
                    T = min(len(fC_arr), len(fD_arr), len(R_arr))
                    if T <= 0:
                        continue
                    tail_len = max(1, int(T * tail_ratio))
                    sl = slice(T - tail_len, T)
                    tail_mean_fC = float(fC_arr[sl].mean())
                    tail_mean_fD = float(fD_arr[sl].mean())
                    tail_mean_R = float(R_arr[sl].mean())

                    writer.writerow([
                        L,
                        f"{r:.4f}",
                        mode,
                        episode_length,
                        num_repeat,
                        tail_ratio,
                        tail_len,
                        tail_mean_fC,
                        tail_mean_fD,
                        tail_mean_R,
                    ])
