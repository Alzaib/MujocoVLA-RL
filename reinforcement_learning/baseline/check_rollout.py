import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # for rollout.py
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout
from eval_bc import run_episode
from bc_model import load_policy
from rollout import run_episode_recorded

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)
env = PickPlaceEnv()
rng = np.random.default_rng(123)

same = 0
for i in range(12):
    layout = sample_layout(rng)
    box, zone = ("red", "green") if i % 2 == 0 else ("white", "pink")
    ok1, d1 = run_episode(env, model, q_mean, q_std, vocab, layout, box, zone, device)
    out = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone, device)
    match = (ok1 == out["success"]) and abs(d1 - out["distance"]) < 1e-6
    same += match
    print(i, box, zone, ok1, round(d1, 4), out["success"], round(out["distance"], 4),
          "MATCH" if match else "DIFFERENT")
print(f"{same}/12 identical")