import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout
from bc_model import load_policy
from rollout import run_episode_recorded
from actor import ResidualActor, make_actor_fn

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
qm, qs = q_mean.cpu().numpy(), q_std.cpu().numpy()
env = PickPlaceEnv()
actor = ResidualActor()
rng = np.random.default_rng(123)

same = 0
for i in range(12):
    layout = sample_layout(rng)
    box, zone = ("red", "green") if i % 2 == 0 else ("white", "pink")
    a = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone, device)
    fn = make_actor_fn(actor, box, zone, qm, qs, explore=False)
    b = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone, device,
                             actor_fn=fn)
    match = a["success"] == b["success"] and abs(a["distance"] - b["distance"]) < 1e-3
    same += match
    print(i, box, zone, a["success"], round(a["distance"], 4),
          b["success"], round(b["distance"], 4), "MATCH" if match else "DIFFERENT")
print(f"{same}/12 match")

# one noisy episode: is the noise really applied?
fn = make_actor_fn(actor, "red", "green", qm, qs, explore=True)
out = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, "red", "green", device,
                           actor_fn=fn)
print("mean |delta| per step:",
      round(float(np.mean([np.abs(s["delta"]).mean() for s in out["steps"]])), 4),
      "(expect about 0.004)")
print("rewards:", out["rewards"])