import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor import ResidualActor, build_obs, OBS_DIM

actor = ResidualActor()
print("parameters:", sum(p.numel() for p in actor.parameters()))

obs = torch.randn(10000, OBS_DIM)

mu = actor.mean(obs)
print("1. largest |mean| at the start:", mu.abs().max().item())

delta, logp = actor.act(obs, explore=True)
print("2. noise std per joint:", delta.std(0).numpy().round(4))

with torch.no_grad():
    actor.net[-1].weight.normal_(0, 5.0)
mu = actor.mean(obs)
print("3. largest |mean| with huge weights:", mu.abs().max().item())

q_mean = np.zeros(7, dtype=np.float32)
q_std = np.ones(7, dtype=np.float32)
x = build_obs(np.arange(7) * 0.1, 3, "white", "pink", 0.04, q_mean, q_std)
print("4. observation size:", tuple(x.shape))
print(x.numpy().round(2))