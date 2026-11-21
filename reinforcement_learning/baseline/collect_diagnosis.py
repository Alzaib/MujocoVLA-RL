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

n, seed = 50, 123                          # the fixed protocol
device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)
env = PickPlaceEnv()
rng = np.random.default_rng(seed)

records, t0 = [], time.time()
for box in BOX_COLORS:
    for zone in ZONE_COLORS:
        for _ in range(n):
            layout = sample_layout(rng)
            out = run_episode_recorded(env, model, q_mean, q_std, vocab,
                                       layout, box, zone, device)
            records.append({
                "box": box, "zone": zone,
                "success": out["success"], "distance": out["distance"],
                "box_start": {c: out["box_start"][c].tolist() for c in BOX_COLORS},
                "zones": {c: out["zones"][c].tolist() for c in ZONE_COLORS},
                "boxes_after_step": [{c: s["boxes"][c].tolist() for c in BOX_COLORS}
                                     for s in out["steps"]],
            })
            if len(records) % 10 == 0:
                wins = sum(r["success"] for r in records)
                print(f"[{len(records)}/300] running {wins}/{len(records)}  "
                      f"{time.time() - t0:.0f}s", flush=True)

out_path = HERE / "results" / "diagnosis_data.json"
with open(out_path, "w") as f:
    json.dump({"seed": seed, "n_per_instruction": n, "episodes": records}, f)
print("saved", out_path)

wins = sum(r["success"] for r in records)
print(f"\noverall: {wins}/{len(records)}")

succ = [r for r in records if r["success"]]
print("\ntarget-box height (m) at the end of each step, successful episodes:")
for k in range(7):
    z = [r["boxes_after_step"][k][r["box"]][2] for r in succ]
    print(f"  step {k}: min {min(z):.3f}  median {np.median(z):.3f}  max {max(z):.3f}")