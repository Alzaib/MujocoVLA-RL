"""collect.py: run the scripted expert and save (image, state, instruction, action) per step."""
import time
from pathlib import Path

import numpy as np

import expert
from env import (PickPlaceEnv, sample_layout, OBJECTS, BOX_COLORS, ZONE_COLORS,
                 GRIPPER_OPEN)


def collect_episode(env, rng):
    """One expert episode. Returns a dict to save, or None if IK failed or the task failed."""
    layout = sample_layout(rng)
    env.reset(layout)
    box = str(rng.choice(BOX_COLORS))
    zone = str(rng.choice(ZONE_COLORS))

    Q = expert.plan(env, box, zone)
    if Q is None:
        return None

    images, qpos, finger = [], [], []

    def on_step(k):                                   # called at the start of each step
        images.append(env.render())
        qpos.append(env.data.qpos[:7].copy())
        finger.append(env.data.qpos[7])

    expert.execute(env, Q, on_step=on_step)

    if not env.is_success(box, zone):
        return None

    target_q, gripper_cmd = expert.step_labels(Q)
    return {
        "images": np.stack(images),                   # (7, H, W, 3) uint8
        "finger": np.array(finger),                   # (7,)  model input
        "qpos": np.stack(qpos),                       # (7, 7) saved for debugging / ablations
        "step": np.arange(7),
        "target_q": target_q,                         # (7, 7)
        "target_gripper": (gripper_cmd == GRIPPER_OPEN).astype(np.float32),   # (7,), 1 = open
        "instruction": f"put the {box} box in the {zone} zone",
        "box_color": box,
        "zone_color": zone,
        "layout": np.stack([layout[c] for c in OBJECTS]),     # (5, 2), rows in OBJECTS order
        "distance": env.distance(box, zone),
    }


def collect(n_episodes, out_file, seed=0, image_size=128):
    """Collect `n_episodes` successful episodes and save them all in one file."""
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = PickPlaceEnv(image_size=image_size)

    episodes, n_failed = [], 0
    t_start = time.perf_counter()
    while len(episodes) < n_episodes:
        episode = collect_episode(env, rng)
        if episode is None:
            n_failed += 1
            continue
        episodes.append(episode)
        print(f"{len(episodes):5d}/{n_episodes}  {episode['instruction']:40s}  "
              f"d={episode['distance']*1000:4.1f} mm  {time.perf_counter() - t_start:6.1f}s", flush=True)

    # stack the list of episodes into one array per field: (N, 7, ...) 
    data = {key: np.stack([ep[key] for ep in episodes]) for key in episodes[0]}
    np.savez_compressed(out_file, **data)
    print(f"saved {n_episodes} episodes to {out_file}, discarded {n_failed} failed, "
          f"in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    collect(2000, "behavioural_cloning/data/train.npz", seed=1)
    collect(200, "behavioural_cloning/data/val.npz", seed=2)
    #collect(2, "behavioural_cloning/data/smoke_test.npz", seed=3)

