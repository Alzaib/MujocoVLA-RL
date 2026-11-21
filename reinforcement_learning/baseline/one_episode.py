import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout, GRIPPER_OPEN, GRIPPER_CLOSED
from bc_model import load_policy, predict

STEP_DURATIONS = [2.0, 2.0, 2.0, 2.5, 2.0, 2.0, 2.0]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)

env = PickPlaceEnv()
layout = sample_layout(np.random.default_rng(123))
box, zone = "red", "green"
instruction = f"put the {box} box in the {zone} zone"

lo, hi = env.model.jnt_range[:7, 0], env.model.jnt_range[:7, 1]

env.reset(layout)
env.run(1.0)

gripper_open = True
for k in range(7):
    image = env.render()
    finger = float(env.data.qpos[7])

    q, g = predict(model, q_mean, q_std, vocab, image, instruction, finger, k, device)
    target = np.clip(q[0], lo, hi)
    env.move_to(target, STEP_DURATIONS[k])

    want_open = bool(g[0])
    if want_open != gripper_open:
        env.run(0.5)
        env.set_gripper(GRIPPER_OPEN if want_open else GRIPPER_CLOSED)
        gripper_open = want_open

    finger_after = float(env.data.qpos[7])
    print(f"step {k}: finger before={finger:.3f} after={finger_after:.3f}  "
          f"plan flags={g.astype(int)}  gripper now {'open' if gripper_open else 'closed'}")

print("success:", env.is_success(box, zone))
print("distance:", env.distance(box, zone))