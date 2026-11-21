"""make_comparison_gif.py - the SAME scene played by BC, BC+RL (actor-critic) and BC+RL (GRPO)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "blackbox"))
from eval_actor import load_actor, play             # also sets up the other import paths

from env import PickPlaceEnv, sample_layout
from bc_model import load_policy
from gif_recorder import Recorder

BC_PATH = HERE / "baseline" / "bc_baseline.pt"


def load_eps(folder):
    return json.load(open(folder / "eval_episodes.json"))["episodes"]


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                # older Pillow
        return ImageFont.load_default()


def find(args, ac_dir, gr_dir, suffix):
    bc = load_eps(ac_dir / ("eval_bc_control" + suffix))
    ac = load_eps(ac_dir / (f"eval_{args.ac_tag}" + suffix))
    gr = load_eps(gr_dir / (f"eval_{args.grpo_tag}" + suffix))
    hits = [i for i in range(len(bc))
            if bc[i]["reward"] == 0 and ac[i]["reward"] == 1.5 and gr[i]["reward"] == 1.5]
    print(f"{len(hits)} scenes where BC never lifts the box but both RL versions succeed")
    for i in hits[:15]:
        print(f"  --index {i}: {bc[i]['box']} -> {bc[i]['zone']}")


def record(env, bc, layout, box, zone, actor, device, every=20):
    fps = 1.0 / (env.model.opt.timestep * every)
    rec = Recorder(env, "unused.gif", every=every, fps=fps)      # we never call rec.save()
    env.on_step = rec
    r = play(env, bc, layout, box, zone, actor, device)
    env.on_step = None
    return [np.asarray(f) for f in rec.frames], bool(r["success"]), fps


def compose(runs, scale, hold=20):
    """runs: list of (label, frames, success). Returns side-by-side PIL frames."""
    f = font(18)
    n = max(len(fr) for _, fr, _ in runs) + hold
    panels = []
    for label, frames, ok in runs:
        out = []
        for t in range(n):
            im = Image.fromarray(frames[min(t, len(frames) - 1)])
            if scale != 1:
                im = im.resize((int(im.width * scale), int(im.height * scale)))
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, im.width, 26], fill=(0, 0, 0))
            d.text((6, 3), label, fill=(255, 255, 255), font=f)
            if t >= len(frames):                                  # this run is over
                d.text((6, im.height - 28), "SUCCESS" if ok else "FAILED",
                       fill=(0, 255, 0) if ok else (255, 60, 60), font=f)
            out.append(np.asarray(im))
        panels.append(out)
    return [Image.fromarray(np.hstack([p[t] for p in panels])) for t in range(n)]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ac_run", default="run1_lr1e3_sig01")
    p.add_argument("--grpo_run", default="grpo_run1")
    p.add_argument("--ac_tag", default="final")
    p.add_argument("--grpo_tag", default="final")
    p.add_argument("--eval_seed", type=int, default=123)
    p.add_argument("--find", action="store_true")                # list candidate scenes
    p.add_argument("--index", type=int, default=None)            # which test episode to render
    p.add_argument("--scale", type=float, default=0.6)           # shrink frames to keep the GIF small
    args = p.parse_args()

    ac_dir = HERE / "blackbox" / "results" / args.ac_run
    gr_dir = HERE / "grpo" / "results" / args.grpo_run
    suffix = "" if args.eval_seed == 123 else f"_seed{args.eval_seed}"

    if args.find:
        find(args, ac_dir, gr_dir, suffix)
        sys.exit()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bc = load_policy(str(BC_PATH), device)
    ep = load_eps(ac_dir / ("eval_bc_control" + suffix))[args.index]
    box, zone = ep["box"], ep["zone"]
    rng = np.random.default_rng(args.eval_seed)                  # replay the evaluation's random stream
    for _ in range(args.index + 1):
        layout = sample_layout(rng)
    print(f"test episode {args.index}: put the {box} box in the {zone} zone")

    env = PickPlaceEnv()
    runs = []
    for label, actor in [("BC only", None),
                         ("BC + RL (actor-critic)", load_actor(ac_dir, args.ac_tag)),
                         ("BC + RL (GRPO)", load_actor(gr_dir, args.grpo_tag))]:
        frames, ok, fps = record(env, bc, layout, box, zone, actor, device)
        print(f"{label:24s}: {'success' if ok else 'failed'}", flush=True)
        runs.append((label, frames, ok))

    imgs = compose(runs, args.scale)
    imgs = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in imgs]
    out_dir = HERE / "comparison_gifs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"seed{args.eval_seed}_idx{args.index}.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    print("saved", out)