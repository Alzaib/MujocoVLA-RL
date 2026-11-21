"""check_val0.py - is BC-alone validation on the 60 tasks stable from run to run?"""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grpo_collect                                   # sets up the import paths

from env import PickPlaceEnv, sample_layout, BOX_COLORS, ZONE_COLORS
from bc_model import load_policy
from rollout import run_episode_recorded
from actor import ResidualActor, make_actor_fn

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE.parent / "baseline" / "bc_baseline.pt"), device)
to_np = lambda x: torch.as_tensor(x).detach().cpu().numpy()
qm, qs = to_np(q_mean), to_np(q_std)
env = PickPlaceEnv()

val_rng = np.random.default_rng(777)                  # same 60 tasks as train.py
val_tasks = [(sample_layout(val_rng),
              BOX_COLORS[val_rng.integers(len(BOX_COLORS))],
              ZONE_COLORS[val_rng.integers(len(ZONE_COLORS))])
             for _ in range(60)]

for rep in range(3):                                  # plain BC, no actor at all
    wins = sum(run_episode_recorded(env, model, q_mean, q_std, vocab, l, b, z, device)["success"]
               for l, b, z in val_tasks)
    print(f"BC alone, pass {rep + 1}: {wins}/60 = {wins / 60:.0%}", flush=True)

actor = ResidualActor(max_delta=0.03, init_sigma=0.01)  # fresh actor: should equal BC
wins = 0
with torch.no_grad():
    for l, b, z in val_tasks:
        fn = make_actor_fn(actor, b, z, qm, qs, explore=False)
        wins += run_episode_recorded(env, model, q_mean, q_std, vocab, l, b, z, device,
                                     actor_fn=fn)["success"]
print(f"zero actor : {wins}/60 = {wins / 60:.0%}")