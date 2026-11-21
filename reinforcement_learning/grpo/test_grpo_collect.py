"""test_grpo_collect.py - play one group of 8 on one scene and look at the output."""
import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grpo_collect import collect_group              # also sets up the other import paths

from env import PickPlaceEnv, sample_layout
from bc_model import load_policy
from actor import ResidualActor

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
to_np = lambda x: torch.as_tensor(x).detach().cpu().numpy()
qm, qs = to_np(q_mean), to_np(q_std)

actor = ResidualActor(max_delta=0.03, init_sigma=0.01)     # fresh actor: starts doing nothing
env = PickPlaceEnv()
layout = sample_layout(np.random.default_rng(0))
box, zone, G = "red", "green", 8

t0 = time.time()
out = collect_group(env, model, q_mean, q_std, vocab, actor, qm, qs, layout, box, zone, G, device)
print(f"time: {time.time() - t0:.1f}s")
print("obs shape   :", tuple(out["obs"].shape), "  (expect 56 x 20)")
print("delta shape :", tuple(out["delta"].shape), "  (expect 56 x 7)")
print("ep_reward   :", out["ep_reward"])
print("success     :", out["success"].astype(int))

# how much do the 8 episodes differ at each step? (spread of BC's waypoint, in scaled units)
spread = [round(float(out["obs"][k::7, :7].std(dim=0).max()), 4) for k in range(7)]
print("BC waypoint spread across the group, step 0..6:", spread)