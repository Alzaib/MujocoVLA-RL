"""grpo_collect.py - play one GROUP of episodes on the same scene and instruction."""
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                                   # rollout.py
sys.path.insert(0, str(HERE.parent / "blackbox"))                      # actor.py
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))    # env.py, bc_model.py

from rollout import run_episode_recorded
from actor import build_obs, make_actor_fn
from env import sample_layout, BOX_COLORS, ZONE_COLORS


def collect_group(env, model, q_mean, q_std, vocab, actor, qm, qs,
                  layout, box, zone, group_size, device="cpu"):
    """Play `group_size` episodes on ONE scene. They differ only through the actor's sampled noise."""
    obs_rows, delta_rows, ep_reward, success = [], [], [], []
    actor_fn = make_actor_fn(actor, box, zone, qm, qs, explore=True)   # explore=True: sample the nudge
    for _ in range(group_size):
        with torch.no_grad():
            ep = run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone,
                                      device, actor_fn=actor_fn)
        for s in ep["steps"]:                       # 7 rows per episode, in step order
            obs_rows.append(build_obs(s["q_pred"][0], s["k"], box, zone, s["finger"], qm, qs))
            delta_rows.append(s["delta"])
        ep_reward.append(float(sum(ep["rewards"])))   # one score per episode
        success.append(bool(ep["success"]))
    return {"obs": torch.stack(obs_rows),                                          # (G*7, 20)
            "delta": torch.as_tensor(np.stack(delta_rows), dtype=torch.float32),   # (G*7, 7)
            "ep_reward": np.array(ep_reward),                                      # (G,)
            "success": np.array(success)}                                          # (G,)

def collect_batch(env, model, q_mean, q_std, vocab, actor, qm, qs, rng,
                  n_groups, group_size, device="cpu"):
    """Several groups, each on its own random scene and random instruction."""
    boxes, zones = list(BOX_COLORS), list(ZONE_COLORS)
    groups = []
    for _ in range(n_groups):
        layout = sample_layout(rng)
        box = boxes[int(rng.integers(len(boxes)))]
        zone = zones[int(rng.integers(len(zones)))]
        groups.append(collect_group(env, model, q_mean, q_std, vocab, actor, qm, qs,
                                    layout, box, zone, group_size, device))
    return groups