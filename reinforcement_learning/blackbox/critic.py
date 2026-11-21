"""critic.py - predicts the expected remaining reward from a situation."""
import torch.nn as nn

from actor import OBS_DIM       # the same 20 numbers the actor reads


class Critic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        """obs: (..., 20) -> one value per situation, shape (...)."""
        return self.net(obs).squeeze(-1)


def rewards_to_returns(rewards):
    """rewards: the 7 per-step rewards of one episode.
    Returns, for each step, the sum of the rewards from that step to the end."""
    returns, total = [], 0.0
    for r in reversed(rewards):
        total += r
        returns.append(total)
    return returns[::-1]