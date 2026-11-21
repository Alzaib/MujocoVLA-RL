"""grpo_update.py - group advantages and one actor update (no critic)."""
import numpy as np
import torch

STEPS = 7                                       # rows per episode


def group_advantages(ep_reward):
    """Compare each episode with its siblings: (reward - group mean) / group std."""
    ep_reward = np.asarray(ep_reward, dtype=np.float64)
    return (ep_reward - ep_reward.mean()) / (ep_reward.std() + 1e-6)   # all equal -> all zero


def make_batch(groups):
    """Stack several collect_group outputs into one training batch."""
    obs = torch.cat([g["obs"] for g in groups])
    delta = torch.cat([g["delta"] for g in groups])
    adv = np.concatenate([np.repeat(group_advantages(g["ep_reward"]), STEPS) for g in groups])
    mixed = float(np.mean([g["ep_reward"].std() > 0 for g in groups]))   # share of groups with variety
    return obs, delta, torch.as_tensor(adv, dtype=torch.float32), mixed


def update_step(actor, actor_opt, obs, delta, advantage):
    log_prob = actor.log_prob(obs, delta)                    # how likely was each nudge?
    loss = -(advantage * log_prob).mean()                    # good episodes up, bad ones down
    actor_opt.zero_grad()
    loss.backward()
    actor_opt.step()
    return {"actor_loss": float(loss.item())}