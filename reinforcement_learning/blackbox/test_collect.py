import sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv
from bc_model import load_policy
from actor import ResidualActor
from rl_collect import collect_batch



device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
qm, qs = q_mean.cpu().numpy(), q_std.cpu().numpy()
env = PickPlaceEnv()
actor = ResidualActor()
rng = np.random.default_rng(2000)         # training seed: different from the test seed 123

t0 = time.time()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
obs, delta, ret, stats = collect_batch(env, model, q_mean, q_std, vocab, actor, qm, qs,
                                       rng, N, device)
print("time:", round(time.time() - t0, 1), "s")
print("obs", tuple(obs.shape), " delta", tuple(delta.shape), " returns", tuple(ret.shape))
print("stats:", stats)
print("returns of the first episode:", ret[:7].tolist())
print("step of the first 7 rows:", obs[:7, 7:14].argmax(1).tolist())
print("mean |delta|:", round(delta.abs().mean().item(), 4))