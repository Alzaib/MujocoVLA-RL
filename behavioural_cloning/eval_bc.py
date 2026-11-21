import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")                      # draw to files only, never open a window
import matplotlib.pyplot as plt

from gif_recorder import Recorder

from env import (PickPlaceEnv, sample_layout, BOX_COLORS, ZONE_COLORS,
                 GRIPPER_OPEN, GRIPPER_CLOSED, X_RANGE, Y_RANGE)
from bc_model import load_policy, predict

# Must match how expert.execute times each step, so the images the network sees
# look like the ones it was trained on.
STEP_DURATIONS = [2.0, 2.0, 2.0, 2.5, 2.0, 2.0, 2.0]


def run_episode(env, model, q_mean, q_std, vocab, layout, box_color, zone_color,
                device="cpu", viewer=None, expert=None):
    """One episode driven by the network. Returns (success, final box-zone distance)."""
    env.reset(layout)
    env.run(1.0, viewer)                                   # settle, as in the expert
    instruction = f"put the {box_color} box in the {zone_color} zone"
    lo, hi = env.model.jnt_range[:7, 0], env.model.jnt_range[:7, 1]
    gripper_open = True

    for k in range(7):
        image = env.render()
        finger = float(env.data.qpos[7])

        if expert is None:
            q, g = predict(model, q_mean, q_std, vocab, image, instruction, finger, k, device)
            q_first, want_open = q[0], bool(g[0])
        else:
            q_first = expert["target_q"][k]
            want_open = bool(expert["target_gripper"][k] > 0.5)

        target = np.clip(q_first, lo, hi)
        env.move_to(target, STEP_DURATIONS[k], viewer)

        if want_open != gripper_open:
            env.run(0.5, viewer)
            env.set_gripper(GRIPPER_OPEN if want_open else GRIPPER_CLOSED, viewer)
            gripper_open = want_open

    ok = bool(env.is_success(box_color, zone_color))
    dist = float(env.distance(box_color, zone_color))
    return ok, dist


# ---------------------------------------------------------------- simulator evaluation
def evaluate(model, q_mean, q_std, vocab, n_per_instruction=10, seed=123, device="cpu",
             out_dir=None, checkpoint=""):
    """Run fresh random layouts for every instruction. Saves results if out_dir is given."""
    env = PickPlaceEnv()
    rng = np.random.default_rng(seed)                      # different from the train/val seeds
    records = []
    for box in BOX_COLORS:
        for zone in ZONE_COLORS:
            wins = 0
            for _ in range(n_per_instruction):
                layout = sample_layout(rng)
                ok, dist = run_episode(env, model, q_mean, q_std, vocab,
                                       layout, box, zone, device)
                wins += ok
                records.append({
                    "box": box, "zone": zone, "success": ok, "distance": dist,
                    "box_xy": [float(v) for v in layout[box]],
                    "zone_xy": [float(v) for v in layout[zone]],
                })
            print(f"{box:6s} -> {zone:5s}: {wins}/{n_per_instruction}")
    overall = np.mean([r["success"] for r in records])
    print(f"overall: {overall:.2%}")

    if out_dir is not None:
        save_eval(records, n_per_instruction, seed, out_dir, checkpoint)
    return records


def save_eval(records, n_per_instruction, seed, out_dir, checkpoint=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "eval_episodes.json", "w") as f:
        json.dump({"checkpoint": checkpoint, "n_per_instruction": n_per_instruction,
                   "seed": seed, "episodes": records}, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    overall = np.mean([r["success"] for r in records])

    # 1. success rate per instruction
    pairs = [(b, z) for b in BOX_COLORS for z in ZONE_COLORS]
    rates, labels = [], []
    for b, z in pairs:
        wins = [r["success"] for r in records if r["box"] == b and r["zone"] == z]
        rates.append(np.mean(wins))
        labels.append(f"{sum(wins)}/{len(wins)}")
    axes[0].bar(range(len(pairs)), rates)
    for i, text in enumerate(labels):
        axes[0].text(i, rates[i] + 0.01, text, ha="center")
    axes[0].set_xticks(range(len(pairs)))
    axes[0].set_xticklabels([f"{b} -> {z}" for b, z in pairs], rotation=30, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("success rate")
    axes[0].set_title(f"Success per instruction (overall {overall:.0%})")

    # 2. how far the box ended from the zone
    axes[1].hist([r["distance"] for r in records], bins=30)
    axes[1].axvline(0.04, color="k", linestyle="--", label="success threshold (4 cm)")
    axes[1].set_xlabel("final box-zone distance (m)")
    axes[1].set_ylabel("episodes")
    axes[1].set_title("Final distance")
    axes[1].legend()

    # 3. where the target box started, success vs failure
    for ok, marker, name in [(True, "o", "success"), (False, "x", "failure")]:
        pts = np.array([r["box_xy"] for r in records if r["success"] == ok])
        if len(pts):
            axes[2].scatter(pts[:, 0], pts[:, 1], marker=marker, label=name)
    axes[2].set_xlim(*X_RANGE)
    axes[2].set_ylim(*Y_RANGE)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    axes[2].set_title("Start position of the target box")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out_dir / "eval_summary.png", dpi=150)
    plt.close(fig)
    print("saved to", out_dir)


# ---------------------------------------------------------------- offline evaluation
def offline_report(model, q_mean, q_std, vocab, data_path="behavioural_cloning/data/val.npz", n_swap=100,
                   swap_zone="pink", device="cpu", out_dir=None, checkpoint=""):
    """Joint error on saved episodes, plus the colour-swap test. No simulator needed."""
    data = dict(np.load(data_path))
    n = len(data["step"])

    # error of the first predicted waypoint vs the expert's target, per episode and step
    errs = np.zeros((n, 7))
    for i in range(n):
        instruction = str(data["instruction"][i])
        for k in range(7):
            q, _ = predict(model, q_mean, q_std, vocab, data["images"][i, k], instruction,
                           float(data["finger"][i, k]), k, device)
            errs[i, k] = np.abs(q[0] - data["target_q"][i, k]).mean()

    per_step = errs.mean(axis=0)
    box_err = {c: float(errs[data["box_color"] == c, 0].mean()) for c in BOX_COLORS}
    zone_err = {c: float(errs[data["zone_color"] == c, 3].mean()) for c in ZONE_COLORS}

    # colour-swap test: same image, different box colour in the instruction
    swap = {}
    for a, b in combinations(BOX_COLORS, 2):
        diffs = []
        for i in range(min(n_swap, n)):
            image = data["images"][i, 0]
            finger = float(data["finger"][i, 0])
            qa, _ = predict(model, q_mean, q_std, vocab, image,
                            f"put the {a} box in the {swap_zone} zone", finger, 0, device)
            qb, _ = predict(model, q_mean, q_std, vocab, image,
                            f"put the {b} box in the {swap_zone} zone", finger, 0, device)
            diffs.append(np.abs(qa[0] - qb[0]).mean())
        swap[f"{a} vs {b}"] = float(np.mean(diffs))

    print("per step (rad):", per_step.round(3), " overall:", errs.mean().round(3))
    print("step 0 by box colour :", {k: round(v, 3) for k, v in box_err.items()})
    print("step 3 by zone colour:", {k: round(v, 3) for k, v in zone_err.items()})
    print("colour swap (rad)    :", {k: round(v, 3) for k, v in swap.items()})

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "offline_step_errors.npy", errs)
        with open(out_dir / "offline_report.json", "w") as f:
            json.dump({"checkpoint": checkpoint, "data": data_path,
                       "per_step": per_step.tolist(), "overall": float(errs.mean()),
                       "step0_by_box": box_err, "step3_by_zone": zone_err,
                       "colour_swap": swap}, f, indent=1)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes = axes.ravel()
        axes[0].bar(range(7), per_step)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("mean abs joint error (rad)")
        axes[0].set_title("Error per step")
        axes[1].bar(list(box_err), list(box_err.values()))
        axes[1].set_ylabel("rad")
        axes[1].set_title("Step 0 error by box colour")
        axes[2].bar(list(zone_err), list(zone_err.values()))
        axes[2].set_ylabel("rad")
        axes[2].set_title("Step 3 error by zone colour")
        axes[3].bar(list(swap), list(swap.values()))
        axes[3].set_ylabel("rad")
        axes[3].set_title("Colour swap: change in prediction")
        plt.tight_layout()
        fig.savefig(out_dir / "offline_report.png", dpi=150)
        plt.close(fig)
        print("saved to", out_dir)

def success_heatmap(folder_path):
    with open(folder_path) as f:
        eps = json.load(f)["episodes"]
    xy = np.array([e["box_xy"] for e in eps])
    ok = np.array([e["success"] for e in eps])

    edges_x = np.linspace(0.30, 0.70, 5)
    edges_y = np.linspace(-0.30, 0.30, 5)
    rate = np.full((4, 4), np.nan)           # rows: x bins, columns: y bins
    count = np.zeros((4, 4), int)
    for i in range(4):
        for j in range(4):
            m = ((xy[:, 0] >= edges_x[i]) & (xy[:, 0] < edges_x[i + 1]) &
                (xy[:, 1] >= edges_y[j]) & (xy[:, 1] < edges_y[j + 1]))
            count[i, j] = m.sum()
            if m.sum():
                rate[i, j] = ok[m].mean()

    plt.imshow(rate.T, origin="lower", extent=[0.3, 0.7, -0.3, 0.3], vmin=0, vmax=1, cmap="viridis")
    for i in range(4):
        for j in range(4):
            plt.text((edges_x[i] + edges_x[i + 1]) / 2, (edges_y[j] + edges_y[j + 1]) / 2,
                    f"{rate[i, j]:.0%}\n(n={count[i, j]})", ha="center", va="center", color="w")
    plt.colorbar(label="success rate")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Success rate by start position of the target box")
    plt.savefig("behavioural_cloning/results/run1_greytable_100ep/success_heatmap.png", dpi=150)
    plt.show()

def watch_one(model, q_mean, q_std, vocab, box="blue", zone="pink", seed=5,
              device="cpu", gif=None, every=20):
    env = PickPlaceEnv()
    layout = sample_layout(np.random.default_rng(seed))
    print(f"instruction: put the {box} box in the {zone} zone")

    if gif:                                              # record, no GUI
        fps = 1.0 / (env.model.opt.timestep * every)     # plays at real speed
        rec = Recorder(env, gif, every=every, fps=fps)
        env.on_step = rec
        ok, dist = run_episode(env, model, q_mean, q_std, vocab, layout, box, zone, device)
        env.on_step = None
        rec.frames += [rec.frames[-1]] * 10              # hold the last frame
        rec.save()
        print("success:", ok, "| box-zone distance:", round(dist, 3))
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        ok, dist = run_episode(env, model, q_mean, q_std, vocab, layout, box, zone, device, viewer)
        print("success:", ok, "| box-zone distance:", round(dist, 3))
        while viewer.is_running():
            time.sleep(0.1)
            break


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = "run1_greytable_100ep"
    checkpoint = "behavioural_cloning/data/bc_best_white_100epochs_greytable.pt"
    out_dir = Path("behavioural_cloning/results") / run_name

    model, q_mean, q_std, vocab = load_policy(checkpoint, device)
    #offline_report(model, q_mean, q_std, vocab, device=device,
    #               out_dir=out_dir, checkpoint=checkpoint)
    #evaluate(model, q_mean, q_std, vocab, n_per_instruction=50, device=device,
    #         out_dir=out_dir, checkpoint=checkpoint)
    import random

    seed = random.randint(0, 1000)
    random.seed(seed)
    print("seed:", seed)

    boxes = ["white", "blue", "red"]
    zones = ["pink", "green"]

    box = random.choice(boxes)
    zone = random.choice(zones)

    print("box:", box)
    print("zone:", zone)


    watch_one(model, q_mean, q_std, vocab, device=device, box=box, zone=zone, seed=seed) 
              #,gif="behavioural_cloning/results/smoke_test.gif")
    #seed: 578, box: blue, zone:pink >> succesful, but funny pickup and dropoff