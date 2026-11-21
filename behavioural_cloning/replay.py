"""replay.py: replay an episode from a packed .npz using expert.execute.

Examples:
    python replay.py                               # random episode, GUI
    python replay.py data/train.npz --index 3      # specific episode, GUI
    python replay.py data/train.npz --index 3 --gif expert.gif   # record a GIF, no GUI
"""
import argparse
import time

import numpy as np
import mujoco.viewer

import expert
from env import PickPlaceEnv, OBJECTS
from gif_recorder import Recorder


def replay(path, index=None, gif=None, every=20):
    data = np.load(path)                               # packed file, loaded lazily
    n = len(data["distance"])
    i = np.random.randint(n) if index is None else index
    print(f"episode {i} of {n}")

    layout = dict(zip(OBJECTS, data["layout"][i]))     # rows are in OBJECTS order
    box, zone = str(data["box_color"][i]), str(data["zone_color"][i])

    q = data["target_q"][i]                            # (7, 7)
    Q = {"above_box": q[0], "grasp": q[1], "above_drop": q[3], "release": q[4]}

    env = PickPlaceEnv()
    env.reset(layout)
    print(data["instruction"][i])

    def report():
        print(f"distance: {env.distance(box, zone) * 1000:.1f} mm "
              f"(saved: {float(data['distance'][i]) * 1000:.1f} mm)")
        print("success:", env.is_success(box, zone))

    if gif:                                            # record, no GUI
        fps = 1.0 / (env.model.opt.timestep * every)   # plays at real speed
        rec = Recorder(env, gif, every=every, fps=fps)
        env.on_step = rec
        expert.execute(env, Q, None)
        env.on_step = None
        rec.frames += [rec.frames[-1]] * 10            # hold the last frame
        rec.save()
        report()
        return

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        expert.execute(env, Q, viewer)
        report()
        while viewer.is_running():                      # keep the window open
            viewer.sync()
            time.sleep(0.05)


# ---- edit these, then just run the file ----
PATH  = "data/train.npz"     
INDEX = 786                  
GIF   = "expert.gif"        
EVERY = 20                   

if __name__ == "__main__":
    replay(PATH, INDEX, GIF, EVERY)
    #replay(path=PATH, index=INDEX, every=EVERY)