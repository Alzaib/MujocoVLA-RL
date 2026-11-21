import sys, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout, BOX_COLORS, ZONE_COLORS
from bc_model import load_policy
from rollout import run_episode_recorded

n, seed = 50, 123
sigmas = [0.01, 0.02, 0.05]                 # noise size in radians, per joint
device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)
env = PickPlaceEnv()

results = {}
for si, sigma in enumerate(sigmas):
    rng = np.random.default_rng(seed)               # same 300 layouts every time
    noise_rng = np.random.default_rng(1000 + si)    # separate generator, so layouts are unaffected
    records, t0 = [], time.time()
    for box in BOX_COLORS:
        for zone in ZONE_COLORS:
            for _ in range(n):
                layout = sample_layout(rng)
                out = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone,
                                           device, noise_std=sigma, noise_rng=noise_rng)
                records.append({"box": box, "zone": zone,
                                "success": out["success"], "distance": out["distance"]})
                if len(records) % 100 == 0:
                    print(f"  sigma={sigma}: {len(records)}/300  {time.time() - t0:.0f}s", flush=True)

    wins = sum(r["success"] for r in records)
    per = {f"{b}-{z}": sum(r["success"] for r in records if r["box"] == b and r["zone"] == z)
           for b in BOX_COLORS for z in ZONE_COLORS}
    print(f"sigma={sigma}: {wins}/300 = {wins / 3:.1f}%   {per}\n", flush=True)
    results[str(sigma)] = {"success": wins, "per_instruction": per, "episodes": records}

with open(HERE / "results" / "noise_only.json", "w") as f:
    json.dump({"seed": seed, "n_per_instruction": n, "results": results}, f)
print("saved results/noise_only.json")