from env import run_baseline_grid



if __name__ == "__main__":
    run_baseline_grid(
    L_list=[25, 30],
    r_list=[2.0, 3.0, 4.0, 4.5],
    alloc_modes=("equal", "contrib"),
    num_repeat=10,
    episode_length=150,
    out_dir="results",
)
    print("hello")