import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actor import OBS_DIM
from critic import Critic, rewards_to_returns

critic = Critic()
print("parameters:", sum(p.numel() for p in critic.parameters()))

obs = torch.randn(5, OBS_DIM)
print("1. output shape for 5 situations:", tuple(critic(obs).shape))

print("2. success      :", rewards_to_returns([0, 0, 0.5, 0, 0, 0, 1.0]))
print("   lifted, drop :", rewards_to_returns([0, 0, 0.5, 0, 0, 0, 0.0]))
print("   never lifted :", rewards_to_returns([0, 0, 0, 0, 0, 0, 0.0]))

# 3. can it learn? Fit one fixed situation to the value 1.5
x = torch.randn(1, OBS_DIM)
opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
for i in range(200):
    loss = ((critic(x) - 1.5) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
print("3. after 200 updates, prediction:", round(critic(x).item(), 3), "(target 1.5)")