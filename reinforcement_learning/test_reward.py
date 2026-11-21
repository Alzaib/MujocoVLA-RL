import json, sys
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reward import episode_rewards

eps = json.load(open(HERE / "baseline" / "results" / "diagnosis_data.json"))["episodes"]

totals = Counter()
for r in eps:
    rewards = episode_rewards(r["boxes_after_step"][2][r["box"]][2], r["success"])
    totals[sum(rewards)] += 1

print("episodes by total reward:", dict(totals))
print("mean reward:", round(sum(k * v for k, v in totals.items()) / len(eps), 3))
print(episode_rewards(0.13, True))
print(episode_rewards(0.13, False))
print(episode_rewards(0.02, False))