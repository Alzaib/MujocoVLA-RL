"""actor.py - the small residual actor (black-box version)."""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "behavioural_cloning"))
from env import BOX_COLORS, ZONE_COLORS

OBS_DIM = 7 + 7 + 3 + 2 + 1      # BC's waypoint, step one-hot, box one-hot, zone one-hot, finger


def build_obs(q_first, step, box, zone, finger, q_mean, q_std):
    """Turn one situation into the 20 numbers the actor reads.
    q_first, q_mean, q_std are numpy arrays of 7 joint values (radians)."""
    x = np.zeros(OBS_DIM, dtype=np.float32)
    x[:7] = (q_first - q_mean) / q_std                 # BC's waypoint, scaled like BC scales it
    x[7 + step] = 1.0                                  # which of the 7 steps
    x[14 + BOX_COLORS.index(box)] = 1.0                # which box
    x[17 + ZONE_COLORS.index(zone)] = 1.0              # which zone
    x[19] = finger * 25.0                              # finger opening (BC scales it the same way)
    return torch.from_numpy(x)


class ResidualActor(nn.Module):
    def __init__(self, max_delta=0.03, init_sigma=0.005, hidden=64):
        super().__init__()
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 7),
        )
        nn.init.zeros_(self.net[-1].weight)            # start by doing nothing:
        nn.init.zeros_(self.net[-1].bias)              # the mean correction is exactly 0
        self.log_std = nn.Parameter(torch.full((7,), math.log(init_sigma)))

    def mean(self, obs):
        """The preferred correction, in radians, never larger than max_delta."""
        return self.max_delta * torch.tanh(self.net(obs))

    def act(self, obs, explore=True):
        """Training: sample a correction. Evaluation: return the mean only."""
        mu = self.mean(obs)
        if not explore:
            return mu, None
        dist = torch.distributions.Normal(mu, self.log_std.exp())
        delta = dist.sample()
        return delta, dist.log_prob(delta).sum(-1)
        
    def log_prob(self, obs, delta):
        """Log-probability of corrections that were already sampled.
        obs: (N, 20), delta: (N, 7) -> (N,)"""
        dist = torch.distributions.Normal(self.mean(obs), self.log_std.exp())
        return dist.log_prob(delta).sum(-1)

def make_actor_fn(actor, box, zone, q_mean, q_std, explore=True):
    """Wrap the actor as a function the rollout can call:
    (BC's first waypoint, step, finger) -> correction (numpy, 7 values)."""
    def actor_fn(q_first, step, finger):
        obs = build_obs(q_first, step, box, zone, finger, q_mean, q_std)
        with torch.no_grad():
            delta, _ = actor.act(obs, explore=explore)
        return delta.numpy()
    return actor_fn