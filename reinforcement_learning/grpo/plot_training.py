"""plot_training.py - training vs validation curves for one RL run."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                                   # draw to a file, no window
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def pick(table, names, what):
    for n in names:
        if n in table:
            return table[n]
    raise KeyError(f"no {what} column; found {list(table)}")


def as_fraction(success, table):
    """Accept success stored as a fraction, a count, or a percent."""
    if success.max() <= 1.0:
        return success
    if "episodes" in table:
        return success / table["episodes"]
    return success / 100.0


def smooth(it, y, w):
    if len(y) < w:
        return it, y
    return it[w - 1:], np.convolve(y, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--window", type=int, default=20)    # smoothing for the noisy training curve
    args = p.parse_args()

    run_dir = HERE / "results" / args.run
    tr = read_csv(run_dir / "train_log.csv")
    va = read_csv(run_dir / "val_log.csv")
    print("train columns:", list(tr))
    print("val columns  :", list(va))

    t_it = pick(tr, ["iter", "iteration"], "iteration")
    t_succ = as_fraction(pick(tr, ["success"], "success"), tr) * 100
    t_rew = pick(tr, ["mean_reward", "reward"], "reward")
    v_it = pick(va, ["iter", "iteration"], "iteration")
    v_succ = as_fraction(pick(va, ["success", "val_success"], "success"), va) * 100
    v_rew = pick(va, ["val_mean_reward", "mean_reward", "reward", "val_reward"], "reward")
    print(f"val success first/last: {v_succ[0]:.0f}% / {v_succ[-1]:.0f}%   (check against your log)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, t_y, v_y, label in [(axes[0], t_succ, v_succ, "success rate (%)"),
                                (axes[1], t_rew, v_rew, "mean reward (max 1.5)")]:
        ax.plot(t_it, t_y, color="tab:blue", alpha=0.2)                       # raw batches
        sx, sy = smooth(t_it, t_y, args.window)
        ax.plot(sx, sy, color="tab:blue", label=f"training (exploring, {args.window}-batch avg)")
        ax.plot(v_it, v_y, "o-", color="tab:orange", label="validation (noise off, 60 episodes)")
        ax.axhline(v_y[0], color="gray", linestyle="--", label="BC alone (iter 0)")
        ax.set_xlabel("training iteration (16 episodes each)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Success rate")
    axes[1].set_title("Reward")
    plt.tight_layout()

    out = run_dir / "plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "training_curves.png", dpi=150)
    print("saved to", out / "training_curves.png")