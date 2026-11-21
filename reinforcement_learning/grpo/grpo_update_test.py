"""test_grpo_update.py - check the advantage maths, then one real update."""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grpo_collect import collect_group
from grpo_update import group_advantages, make_batch, update_step

from env import PickPlaceEnv, sample_layout
from bc_model import load_policy
from actor import ResidualActor

# 1. the numbers from our example group
r = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 0.5, 0.0, 1.5])
print("advantages:", group_advantages(r).round(2))
print("sum (should be ~0):", round(float(group_advantages(r).sum()), 6))

# 2. a group where everything is the same -> no signal
print("all-same group:", group_advantages(np.array([1.5] * 8)).round(2))

# 3. one real update on a real group
device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
to_np = lambda x: torch.as_tensor(x).detach().cpu().numpy()
qm, qs = to_np(q_mean), to_np(q_std)
actor = ResidualActor(max_delta=0.03, init_sigma=0.01)
opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
env = PickPlaceEnv()
layout = sample_layout(np.random.default_rng(0))
group = collect_group(env, model, q_mean, q_std, vocab, actor, qm, qs, layout, "red", "green", 8, device)

obs, delta, adv, mixed = make_batch([group])
print("batch:", tuple(obs.shape), tuple(delta.shape), tuple(adv.shape), "| mixed groups:", mixed)
before = actor.mean(obs).abs().mean().item()
info = update_step(actor, opt, obs, delta, adv)
after = actor.mean(obs).abs().mean().item()
print("loss:", round(info["actor_loss"], 4), "| mean|mu| before/after:", round(before, 5), "/", round(after, 5))