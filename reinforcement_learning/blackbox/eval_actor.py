"""eval_actor.py - watch one scene (BC vs RL), or evaluate the trained actor headless."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import time
import mujoco.viewer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                  # actor.py
sys.path.insert(0, str(HERE.parent))                           # rollout.py
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout, BOX_COLORS, ZONE_COLORS
from bc_model import load_policy
from eval_bc import save_eval
from gif_recorder import Recorder
from rollout import run_episode_recorded
from actor import ResidualActor, make_actor_fn

BC_PATH = HERE.parent / "baseline" / "bc_baseline.pt"


def load_actor(run_dir, tag):
    cfg = json.load(open(run_dir / "config.json"))
    actor = ResidualActor(max_delta=cfg["max_delta"], init_sigma=cfg["init_sigma"])
    actor.load_state_dict(torch.load(run_dir / f"actor_{tag}.pt"))
    actor.eval()
    return actor

def to_np(x):
    return torch.as_tensor(x).detach().cpu().numpy()

def play(env, bc, layout, box, zone, actor, device, viewer=None):
    model, qm, qs, vocab = bc
    fn = None if actor is None else make_actor_fn(actor, box, zone, to_np(qm), to_np(qs), explore=False)
    with torch.no_grad():
        return run_episode_recorded(env, model, qm, qs, vocab, layout, box, zone,
                                    device, actor_fn=fn, viewer=viewer)


def watch(args, bc, actor, out_dir, device):
    seed = args.seed if args.seed is not None else random.randint(0, 1000)
    rnd = random.Random(seed)
    box = args.box or rnd.choice(list(BOX_COLORS))
    zone = args.zone or rnd.choice(list(ZONE_COLORS))
    print(f"seed {seed} | put the {box} box in the {zone} zone")
    env = PickPlaceEnv()
    layout = sample_layout(np.random.default_rng(seed))          # same scene for both runs
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, a in [("bc", None), ("rl", actor)]:
        every = 20
        rec = Recorder(env, str(out_dir / f"seed{seed}_{name}.gif"), every=every,
                       fps=1.0 / (env.model.opt.timestep * every))
        env.on_step = rec
        r = play(env, bc, layout, box, zone, a, device)
        env.on_step = None
        rec.frames += [rec.frames[-1]] * 10
        rec.save()
        print(f"{name}: success {r['success']} | distance {r['distance']:.3f}")

def view(args, bc, actor, device):
    seed = args.seed if args.seed is not None else random.randint(0, 1000)
    rnd = random.Random(seed)
    box = args.box or rnd.choice(list(BOX_COLORS))
    zone = args.zone or rnd.choice(list(ZONE_COLORS))
    print(f"seed {seed} | put the {box} box in the {zone} zone")
    env = PickPlaceEnv()
    layout = sample_layout(np.random.default_rng(seed))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for name, a in [("BC", None), ("BC + RL", actor)]:
            print(f"--- {name} ---")
            r = play(env, bc, layout, box, zone, a, device, viewer)
            print(f"{name}: success {r['success']} | distance {r['distance']:.3f}")
            time.sleep(2)                       # short pause before the next run
        print("close the window to exit")
        while viewer.is_running():
            time.sleep(0.1)

def evaluate(args, bc, actor, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=False)                  # never overwrite
    env = PickPlaceEnv()
    rng = np.random.default_rng(args.eval_seed)              # was 123                           # same test layouts as BC eval
    records = []
    for box in BOX_COLORS:
        for zone in ZONE_COLORS:
            wins = 0
            for _ in range(args.n):
                layout = sample_layout(rng)
                r = play(env, bc, layout, box, zone, actor, device)
                wins += r["success"]
                records.append({"box": box, "zone": zone, "success": r["success"],
                                "distance": r["distance"],
                                "box_xy": [float(v) for v in layout[box]],
                                "zone_xy": [float(v) for v in layout[zone]],
                                "reward": float(sum(r["rewards"]))})
            print(f"{box:6s} -> {zone:5s}: {wins}/{args.n}")
    print(f"overall: {np.mean([r['success'] for r in records]):.2%}")
    save_eval(records, args.n, args.eval_seed, out_dir, checkpoint=f"{args.run}/{args.tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--tag", default="final")                     # which actor checkpoint
    p.add_argument("--mode", choices=["watch", "view", "eval"], required=True)
    p.add_argument("--bc_only", action="store_true")             # control: BC with the same code
    p.add_argument("--n", type=int, default=50)                  # episodes per instruction
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--box", default=None)
    p.add_argument("--zone", default=None)
    p.add_argument("--eval_seed", type=int, default=123)         # seed for the test layouts
    p.add_argument("--method", default="blackbox", choices=["blackbox", "grpo"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = HERE.parent / args.method / "results" / args.run
    bc = load_policy(str(BC_PATH), device)
    actor = None if args.bc_only else load_actor(run_dir, args.tag)

    if args.mode == "view":
        view(args, bc, actor, device)
    elif args.mode == "watch":
        watch(args, bc, actor, run_dir / "watch", device)
    else:
        suffix = "" if args.eval_seed == 123 else f"_seed{args.eval_seed}"
        name = ("eval_bc_control" if args.bc_only else f"eval_{args.tag}") + suffix
        evaluate(args, bc, actor, run_dir / name, device)