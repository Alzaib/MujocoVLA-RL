"""collect.py - run a batch of episodes with the current actor and keep what the update needs."""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import sample_layout, BOX_COLORS, ZONE_COLORS
from rollout import run_episode_recorded
from actor import build_obs, make_actor_fn
from critic import rewards_to_returns


def collect_batch(env, model, q_mean, q_std, vocab, actor, qm, qs, rng, n_episodes, device):
    """Returns obs (N,20), delta (N,7), returns (N,), stats; N = 7 * n_episodes.
    qm, qs: BC's q_mean and q_std as numpy arrays."""
    obs_list, delta_list, return_list = [], [], []
    wins, total_reward = 0, 0.0

    for _ in range(n_episodes):
        layout = sample_layout(rng)
        box = BOX_COLORS[rng.integers(len(BOX_COLORS))]      # a random instruction
        zone = ZONE_COLORS[rng.integers(len(ZONE_COLORS))]
        fn = make_actor_fn(actor, box, zone, qm, qs, explore=True)
        out = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone,
                                   device, actor_fn=fn)

        returns = rewards_to_returns(out["rewards"])
        for s, G in zip(out["steps"], returns):
            # rebuild exactly what the actor saw at this step
            obs_list.append(build_obs(s["q_pred"][0], s["k"], box, zone, s["finger"], qm, qs))
            delta_list.append(torch.from_numpy(s["delta"]))
            return_list.append(G)

        wins += out["success"]
        total_reward += sum(out["rewards"])

    stats = {"episodes": n_episodes, "success": wins,
             "mean_reward": round(total_reward / n_episodes, 3)}
    return (torch.stack(obs_list), torch.stack(delta_list),
            torch.tensor(return_list, dtype=torch.float32), stats)