import sys
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "behavioural_cloning"))

from env import PickPlaceEnv, sample_layout
from bc_model import load_policy, predict

device = "cuda" if torch.cuda.is_available() else "cpu"
model, q_mean, q_std, vocab = load_policy(str(HERE / "bc_baseline.pt"), device)

env = PickPlaceEnv()
rng = np.random.default_rng(123)
layout = sample_layout(rng)

env.reset(layout)
env.run(1.0)                                  # let things settle

image = env.render()
finger = float(env.data.qpos[7])
imageio.imwrite(str(HERE / "results" / "first_observation.png"), image)

instruction = "put the red box in the green zone"
q, gripper_open = predict(model, q_mean, q_std, vocab, image, instruction, finger, 0, device)

print("image:", image.shape, image.dtype)
print("finger opening:", round(finger, 4))
print("q shape:", q.shape)
print("=====")
print("first waypoint (rad):", q[0].round(2))
print("second waypoint (rad):", q[1].round(2))
print("third waypoint (rad):", q[2].round(2))
print("forth waypoint (rad):", q[3].round(2))
print("Only first waypoint is executed by the BC")
print("gripper open flags:", gripper_open)