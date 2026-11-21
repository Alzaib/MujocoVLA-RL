"""rollout.py - one episode with a BC policy (optionally corrected), recording every step."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "behavioural_cloning"))

from env import GRIPPER_OPEN, GRIPPER_CLOSED, BOX_COLORS, ZONE_COLORS
from bc_model import predict
from reward import episode_rewards

STEP_DURATIONS = [2.0, 2.0, 2.0, 2.5, 2.0, 2.0, 2.0]


def run_episode_recorded(env, model, q_mean, q_std, vocab, layout, box, zone, device="cpu",
                         noise_std=0.0, noise_rng=None, actor_fn=None, viewer=None):
    env.reset(layout)
    env.run(1.0, viewer)
    instruction = f"put the {box} box in the {zone} zone"
    lo, hi = env.model.jnt_range[:7, 0], env.model.jnt_range[:7, 1]
    gripper_open = True
    steps = []

    box_start = {c: env.box_pos(c) for c in BOX_COLORS}
    zones = {c: env.zone_pos(c) for c in ZONE_COLORS}

    for k in range(7):
        image = env.render()
        finger = float(env.data.qpos[7])
        q, g = predict(model, q_mean, q_std, vocab, image, instruction, finger, k, device)

        # the correction added to BC's first waypoint
        if actor_fn is not None:
            delta = actor_fn(q[0], k, finger)
        elif noise_std > 0:
            delta = noise_rng.normal(0.0, noise_std, size=7)
        else:
            delta = np.zeros(7)

        target = np.clip(q[0] + delta, lo, hi)
        env.move_to(target, STEP_DURATIONS[k], viewer)
        want_open = bool(g[0])
        if want_open != gripper_open:
            env.run(0.5, viewer); env.set_gripper(GRIPPER_OPEN if want_open else GRIPPER_CLOSED, viewer)
            gripper_open = want_open

        steps.append({"k": k, "image": image, "finger": finger,
                      "q_pred": q.copy(), "g_pred": g.copy(), "q_exec": target.copy(),
                      "delta": np.asarray(delta, dtype=np.float32).copy(),
                      "boxes": {c: env.box_pos(c) for c in BOX_COLORS}})

    success = bool(env.is_success(box, zone))
    return {"success": success,
            "distance": float(env.distance(box, zone)),
            "instruction": instruction, "steps": steps,
            "box_start": box_start, "zones": zones,
            "rewards": episode_rewards(steps[2]["boxes"][box][2], success)}