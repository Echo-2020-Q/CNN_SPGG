"""
Simple handcrafted allocation baselines for the spatial public-goods game.

三种手工分配策略：
- labor:   按投入比例分配（按劳分配）
- equal:   小组内均分（平均分配）
- hybrid:  w * labor + (1-w) * equal（混合分配）

本脚本可扫描多个 (L, r)，记录随时间的合作率 f_C，并汇报后 10% 步的稳态平均 f_C。
输出：每次运行都会在 baseline_results/<时间+主要参数>/ 下保存：
- summary.csv：不同 (L, r, allocation) 的稳态 f_C 等指标
- traj_*.csv：平均 f_C 随 t 的轨迹
- traj_*.png：若安装 matplotlib，则绘制对应的 f_C 曲线
"""

from __future__ import annotations

import argparse
import csv
import datetime
import dataclasses
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from env import PublicGoodsEnv


@dataclass
class BaselineConfig:
    # sweep targets
    L_list: Tuple[int, ...] = (32,)
    r_list: Tuple[float, ...] = (3.5,)

    # allocation rule: labor / equal / hybrid （可选单个或 allocations 列表）
    allocation: str = "labor"
    allocations: Optional[Tuple[str, ...]] = None  # 若设置，则按列表循环多个分配模式
    hybrid_w: float = 0.5

    # env settings
    episode_length: int = 150
    episodes: int = 5
    R_decay: float = 0.10
    coop_cost: float = 5.0
    initial_R: float = 50.0
    use_cumulative_planner_reward: bool = False
    investment_mode: str = "fixed"
    investment_tau: float = 0.5
    investment_cmax: float = 45.0
    P_max: float = 50.0
    init_coop_frac: Optional[float] = 0.5  # None 表示每次 reset 随机采样 f_C∈[0,1]
    init_mode: str = "iid"  # iid / exact_frac

    # logging
    save_dir: str = os.path.join(os.path.dirname(__file__), "baseline_results")
    save_traj: bool = True  # save mean f_C trajectory per (L, r)

    seed: Optional[int] = None


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _compute_agent_investment(env: PublicGoodsEnv) -> np.ndarray:
    """
    Reproduce the env's investment calculation so the handcrafted allocation
    matches the actual contributions used inside env.step.
    Returns invest_agent with shape (L, L).

    复用 env 的投入逻辑，确保手工分配用到的贡献量与 env.step 内一致。
    """
    # mirror the clipping and invest rules from env.step
    if env.investment_mode == "fixed_add_proportion":
        steady_max_R = (float(env.r) - 1.0) * (env.coop_cost + env.invest_cmax) / max(
            1e-8, 1.0 - env.R_decay
        )
    elif env.investment_mode == "fixed_add_proportion_poolmax":
        steady_max_R = max(
            (env.P_max + env.coop_cost * (env.invest_tau - 1.0))
            / (env.R_decay + env.invest_tau + 1e-8),
            env.coop_cost * 4.0,
        )
    else:
        steady_max_R = 5.0 * max(float(env.r) - 1.0, 0.0) / (env.R_decay + 1e-8)
    R_cap = max(steady_max_R * 20.0, env.coop_cost * 20.0)
    R_clipped = np.clip(env.R, 0.0, R_cap)

    can_cooperate = (env.strategy == 1) & (R_clipped >= env.coop_cost)
    if env.investment_mode == "fixed_add_proportion":
        invest_extra = np.maximum(0.0, R_clipped - env.coop_cost)
        invest_agent = env.coop_cost + env.invest_cmax * np.tanh(
            (env.invest_tau * invest_extra) / max(1e-8, env.invest_cmax)
        )
        invest_agent = np.where(can_cooperate, invest_agent, 0.0).astype(np.float32)
    elif env.investment_mode == "fixed_add_proportion_poolmax":
        invest_extra = np.maximum(0.0, R_clipped - env.coop_cost)
        invest_agent = env.coop_cost + env.invest_tau * invest_extra
        invest_agent = np.where(can_cooperate, invest_agent, 0.0).astype(np.float32)
    else:
        invest_agent = can_cooperate.astype(np.float32)
    return invest_agent


def _compute_role_contributions(env: PublicGoodsEnv) -> np.ndarray:
    """
    Returns per-group contributions for roles (mid, up, down, left, right),
    shape (L, L, 5), consistent with env.step's roll layout.

    将个体投入 roll 成小组视角，顺序为 (mid, up, down, left, right)。
    """
    invest_agent = _compute_agent_investment(env)
    mid_inv = invest_agent.astype(np.float32)
    up_inv = np.roll(invest_agent, shift=1, axis=0).astype(np.float32)
    down_inv = np.roll(invest_agent, shift=-1, axis=0).astype(np.float32)
    left_inv = np.roll(invest_agent, shift=1, axis=1).astype(np.float32)
    right_inv = np.roll(invest_agent, shift=-1, axis=1).astype(np.float32)
    return np.stack([mid_inv, up_inv, down_inv, left_inv, right_inv], axis=-1)


def build_pi_field(env: PublicGoodsEnv, mode: str, hybrid_w: float = 0.5) -> np.ndarray:
    """
    Construct a (L, L, 5) allocation tensor based on the chosen rule.

    按分配模式构造 (L, L, 5) 的 π_field；当组内无人投入时退化为均分。
    """
    mode = mode.strip().lower()
    hybrid_w = float(np.clip(hybrid_w, 0.0, 1.0))
    L = env.L
    uniform = np.full((L, L, 5), 1.0 / 5.0, dtype=np.float32)

    if mode == "equal":
        return uniform

    contrib = _compute_role_contributions(env)  # (L, L, 5)
    contrib_sum = contrib.sum(axis=-1, keepdims=True)
    # 先安全除法，再对无投入的小组回退到均分，避免维度不匹配
    labor_pi = contrib / np.clip(contrib_sum, 1e-8, None)
    mask_valid = (contrib_sum[..., 0] > 1e-8)  # (L, L)
    labor_pi = labor_pi.astype(np.float32)
    labor_pi[~mask_valid] = uniform[~mask_valid]

    if mode == "labor":
        return labor_pi
    if mode == "hybrid":
        return (hybrid_w * labor_pi + (1.0 - hybrid_w) * uniform).astype(np.float32)
    raise ValueError(f"Unknown allocation mode: {mode}")


def run_single_episode(env: PublicGoodsEnv, cfg: BaselineConfig) -> List[float]:
    """
    Run one episode and return the cooperation rate trajectory (list of f_C).

    单条轨迹：每步用手工策略分配公共池，记录 f_C。
    """
    state = env.reset()
    _ = state  # state is unused; allocation relies on env internals
    traj_fC: List[float] = []
    done = False
    while not done:
        pi_field = build_pi_field(env, cfg.allocation, cfg.hybrid_w)
        _, _, done, info = env.step(pi_field)
        traj_fC.append(float(info.get("f_C", 0.0)))
    return traj_fC


def analyze_fC(trajs: List[List[float]], tail_ratio: float = 0.1) -> Tuple[np.ndarray, float]:
    """
    Compute per-step mean f_C and steady-state mean (last tail_ratio of steps).
    Assumes all trajectories share the same length.

    计算平均轨迹与稳态平均（尾部比例 tail_ratio）。
    """
    if not trajs:
        return np.array([], dtype=np.float32), float("nan")
    arr = np.asarray(trajs, dtype=np.float32)
    mean_traj = arr.mean(axis=0)
    tail_len = max(1, int(len(mean_traj) * tail_ratio))
    steady_mean = float(mean_traj[-tail_len:].mean())
    return mean_traj, steady_mean


def run_baseline_for_combo(
    L: int,
    r: float,
    cfg: BaselineConfig,
) -> Tuple[np.ndarray, float]:
    env = PublicGoodsEnv(
        L=L,
        r=r,
        episode_length=cfg.episode_length,
        use_cumulative_planner_reward=cfg.use_cumulative_planner_reward,
        R_decay=cfg.R_decay,
        coop_cost=cfg.coop_cost,
        initial_R=cfg.initial_R,
        investment_mode=cfg.investment_mode,
        investment_tau=cfg.investment_tau,
        investment_cmax=cfg.investment_cmax,
        P_max=cfg.P_max,
        init_coop_frac=cfg.init_coop_frac,
        init_mode=cfg.init_mode,
    )
    trajs = [run_single_episode(env, cfg) for _ in range(cfg.episodes)]
    mean_traj, steady_mean = analyze_fC(trajs, tail_ratio=0.10)
    return mean_traj, steady_mean


def save_traj_csv(path: str, mean_traj: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "mean_fC"])
        for t, val in enumerate(mean_traj, start=1):
            writer.writerow([t, float(val)])


def save_summary_csv(path: str, rows: List[dict]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sweep(
    L_list: Iterable[int],
    r_list: Iterable[float],
    cfg: BaselineConfig,
):
    """
    扫描多组 (L, r)，保存轨迹与摘要。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 以时间戳 + 主要参数命名本次运行的输出目录
    run_name = (
        f"{timestamp}"
        f"_L{'-'.join(str(x) for x in L_list)}"
        f"_r{'-'.join(str(x) for x in r_list)}"
        f"_alloc{'-'.join(cfg.allocations or (cfg.allocation,))}"
        f"_w{cfg.hybrid_w}"
    )
    run_dir = os.path.join(cfg.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    summary_rows: List[dict] = []

    allocations = cfg.allocations or (cfg.allocation,)

    for alloc in allocations:
        cfg_all = dataclasses.replace(cfg, allocation=alloc, allocations=None)
        for L in L_list:
            for r in r_list:
                mean_traj, steady_mean = run_baseline_for_combo(L, r, cfg_all)
                summary_rows.append(
                    {
                        "L": L,
                        "r": r,
                        "allocation": alloc,
                        "hybrid_w": cfg.hybrid_w,
                        "episode_length": cfg.episode_length,
                        "episodes": cfg.episodes,
                        "steady_mean_fC": steady_mean,
                        "final_fC": float(mean_traj[-1]) if mean_traj.size > 0 else float("nan"),
                    }
                )
                if cfg.save_traj and mean_traj.size > 0:
                    traj_basename = f"traj_L{L}_r{r:g}_{alloc}_w{cfg.hybrid_w}.csv"
                    traj_path = os.path.join(run_dir, traj_basename)
                    save_traj_csv(traj_path, mean_traj)
                    print(f"[Baseline] Saved trajectory to {traj_path}")

                    if plt is not None:
                        plt.figure()
                        xs = np.linspace(1, cfg.episode_length, num=len(mean_traj))
                        plt.plot(xs, mean_traj, label="mean f_C")
                        plt.xlabel("t")
                        plt.ylabel("f_C")
                        plt.title(f"L={L}, r={r}, alloc={alloc}, w={cfg.hybrid_w}")
                        plt.grid(True)
                        plt.ylim(0.0, 1.0)
                        plt.xlim(0.0, cfg.episode_length)
                        plt.tight_layout()
                        png_path = os.path.join(run_dir, traj_basename.replace(".csv", ".png"))
                        plt.savefig(png_path)
                        plt.close()
                        print(f"[Baseline] Saved plot to {png_path}")

    summary_path = os.path.join(run_dir, "summary.csv")
    save_summary_csv(summary_path, summary_rows)
    print(f"[Baseline] Summary saved to {summary_path}")
    for row in summary_rows:
        print(
            f"L={row['L']}, r={row['r']}: steady_mean_fC={row['steady_mean_fC']:.4f}, "
            f"final_fC={row['final_fC']:.4f}"
        )


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(description="Handcrafted allocation baselines for PublicGoodsEnv.")
    parser.add_argument("--L_list", type=int, nargs="+", default=[32], help="List of grid sizes L to test.")
    parser.add_argument("--r_list", type=float, nargs="+", default=[3.5], help="List of r values to test.")
    parser.add_argument("--allocation", choices=["labor", "equal", "hybrid"], default="labor", help="单一分配模式。")
    parser.add_argument(
        "--allocation_list",
        choices=["labor", "equal", "hybrid"],
        nargs="+",
        help="若提供，则按列表依次跑多种分配模式。",
    )
    parser.add_argument("--hybrid_w", type=float, default=0.5, help="Weight for labor part in hybrid mode.")
    parser.add_argument("--episode_length", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--R_decay", type=float, default=0.10)
    parser.add_argument("--coop_cost", type=float, default=5.0)
    parser.add_argument("--initial_R", type=float, default=50.0)
    parser.add_argument("--investment_mode", choices=["fixed", "fixed_add_proportion", "fixed_add_proportion_poolmax"], default="fixed")
    parser.add_argument("--investment_tau", type=float, default=0.5)
    parser.add_argument("--investment_cmax", type=float, default=45.0)
    parser.add_argument("--P_max", type=float, default=50.0, help="公共池上限（适用于 fixed_add_proportion_poolmax）")
    parser.add_argument(
        "--init_coop_frac",
        type=float,
        default=0.5,
        help="初始合作率 f_C；设为负值可表示每次 reset 随机采样 [0,1]。",
    )
    parser.add_argument(
        "--init_mode",
        choices=["iid", "exact_frac"],
        default="iid",
        help="iid: 独立伯努利；exact_frac: 精确比例（四舍五入后打乱位置）。",
    )
    parser.add_argument("--no_save_traj", action="store_true", help="Disable saving f_C trajectories.")
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.path.dirname(__file__), "baseline_results"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = BaselineConfig(
        L_list=tuple(args.L_list),
        r_list=tuple(args.r_list),
        allocation=args.allocation,
        allocations=tuple(args.allocation_list) if args.allocation_list else None,
        hybrid_w=args.hybrid_w,
        episode_length=args.episode_length,
        episodes=args.episodes,
        R_decay=args.R_decay,
        coop_cost=args.coop_cost,
        initial_R=args.initial_R,
        use_cumulative_planner_reward=False,
        investment_mode=args.investment_mode,
        investment_tau=args.investment_tau,
        investment_cmax=args.investment_cmax,
        P_max=args.P_max,
        init_coop_frac=None if args.init_coop_frac is not None and args.init_coop_frac < 0 else args.init_coop_frac,
        init_mode=args.init_mode,
        save_dir=args.save_dir,
        save_traj=not args.no_save_traj,
        seed=args.seed,
    )
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)
    sweep(cfg.L_list, cfg.r_list, cfg)


if __name__ == "__main__":
    # ========== 便捷运行开关（IDE 直接 Shift+F10 无需命令行参数） ==========
    USE_MANUAL_CONFIG = True  # 设为 True 时使用下方手动配置；False 时走命令行参数

    if USE_MANUAL_CONFIG:
        # 手动配置区：可在此修改所有参数后直接运行
        manual_cfg = BaselineConfig(
            L_list=(15, 30, 50),            # 待测试的 L 列表
            r_list=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5),           # 待测试的 r 列表
            allocation="labor",      # 单一分配模式（若 allocations 为空则使用此项）
            allocations=("labor","equal"),  # 多个分配模式列表，例如 ("labor", "equal", "hybrid")
            hybrid_w=0.5,            # hybrid 权重（仅 hybrid 生效）
            episode_length=500,      # 每个 episode 的步数
            episodes=5,              # 同一 (L,r) 的重复次数
            R_decay=0.10,
            coop_cost=5.0,
            initial_R=20.0,
            use_cumulative_planner_reward=False,
            investment_mode="fixed_add_proportion_poolmax",  # fixed / fixed_add_proportion / fixed_add_proportion_poolmax
            investment_tau=0.5,
            investment_cmax=45.0,
            P_max=100.0,
            init_coop_frac=0.5,    # 初始合作率 f_C（None 表示每次 reset 随机采样）
            init_mode="exact_frac",
            save_dir=os.path.join(os.path.dirname(__file__), "baseline_results"),
            save_traj=True,           # 是否保存 f_C 轨迹 CSV
            seed=42,                  # 随机种子（None 表示不设）
        )
        set_seed(manual_cfg.seed)
        sweep(manual_cfg.L_list, manual_cfg.r_list, manual_cfg)
    else:
        # 使用命令行参数运行，例如：
        # python Baseline.py --allocation hybrid --hybrid_w 0.7 --L_list 25 32 --r_list 2.0 3.5
        main()
