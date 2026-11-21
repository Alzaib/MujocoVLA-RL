import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor import ResidualActor
from critic import Critic
from update import update_step

torch.manual_seed(0)
actor, critic = ResidualActor(), Critic()
actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-2)
critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-2)

x = torch.randn(1, 20).repeat(256, 1)        # the same situation, 256 times

for it in range(151):
    with torch.no_grad():
        delta, _ = actor.act(x, explore=True)
    returns = (delta[:, 0] > 0).float()      # made-up reward: 1 if joint 0's correction is positive
    update_step(actor, critic, actor_opt, critic_opt, x, delta, returns)
    if it % 30 == 0:
        mu = actor.mean(x[:1])[0]
        print(f"iter {it:3d}: joint0 mean = {mu[0].item():+.4f}   joint1 mean = {mu[1].item():+.4f}"
              f"   critic value = {critic(x[:1]).item():.2f}")