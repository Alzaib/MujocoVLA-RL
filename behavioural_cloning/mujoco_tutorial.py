import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import ik

np.set_printoptions(precision=4, suppress=True)

#custom scene with panda robot, table, cubes and drop zones
SCENE = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "franka_emika_panda" / "panda_pick_scene.xml"

#load and the models from the scene
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0

# Joint targets from the IK run (rad) - used for testing only
Q_ABOVE_BOX_FK  = np.array([ 0.1959, 0.2747, 0.1810, -2.2644, -0.0855, 2.5331, -0.3448])
Q_GRASP_FK      = np.array([ 0.2422, 0.5641, 0.1174, -2.2182, -0.1752, 2.7740, -0.2801])
Q_ABOVE_DROP_FK = np.array([-0.4707, 0.2712, 0.0886, -2.2650, -0.0416, 2.5348, -1.1365])
Q_RELEASE_FK    = np.array([-0.4506, 0.5344, 0.0604, -2.2284, -0.0828, 2.7608, -1.1069])


TABLE_X = (-0.2, 0.8)          # for reference: the table's extent
TABLE_Y = (-0.4, 0.4)
X_RANGE = (0.30, 0.70)         # sampling region, inside the table with margin
Y_RANGE = (-0.30, 0.30)

OBJECTS = {"red": "box", "blue": "box", "white": "box", "green": "zone", "pink": "zone"}
RADIUS = {"box": 0.05, "zone": 0.06}      # keep-out radius per object type

def sample_layout(rng, max_tries=1000):
    placed = {}
    for name, kind in OBJECTS.items():
        for _ in range(max_tries):
            xy = np.array([rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE)])
            ok = all(np.linalg.norm(xy - p) >= RADIUS[kind] + RADIUS[OBJECTS[other]]
                     for other, p in placed.items())
            if ok:
                placed[name] = xy
                break
        else:
            raise RuntimeError(f"could not place {name} after {max_tries} tries")
    return placed

rng = np.random.default_rng()   # fixed seed means the same "random" positions each run

def sample_box_xy():
    x = rng.uniform(0.2, 0.8)
    y = rng.uniform(0.1, 0.4)
    print("random x, y: ", x, y)
    return (x, y)

def reset(box_xy_red=(0.5, 0.2), 
          box_xy_blue=(0.4, 0.1),
          box_xy_white=(0.6, 0.0),
          green_zone=(0.6, 0.0),
          pink_zone=(0.5, -0.2)):
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_Q
    data.qpos[7:9] = 0.04
    data.ctrl[:7] = HOME_Q
    data.ctrl[7] = GRIPPER_OPEN
    adr = model.joint("red_box_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [box_xy_red[0], box_xy_red[1], 0.02, 1, 0, 0, 0]
    adr = model.joint("blue_box_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [box_xy_blue[0], box_xy_blue[1], 0.02, 1, 0, 0, 0]
    adr = model.joint("white_box_joint").qposadr[0]
    data.qpos[adr:adr + 7] = [box_xy_white[0], box_xy_white[1], 0.02, 1, 0, 0, 0]
    green_id = model.body_mocapid[model.body("green_zone").id]
    pink_id  = model.body_mocapid[model.body("pink_zone").id]
    data.mocap_pos[green_id] = [green_zone[0], green_zone[1], 0.0005]
    data.mocap_pos[pink_id]  = [pink_zone[0], pink_zone[1], 0.0005]
    mujoco.mj_forward(model, data)     # refresh the derived positions, as with the boxes
    mujoco.mj_forward(model, data)


def run(duration, viewer=None):
    """Let the physics run for `duration` seconds of simulated time."""
    t0 = data.time
    n = 0
    while data.time - t0 < duration and (viewer is None or viewer.is_running()):
        mujoco.mj_step(model, data)
        n += 1
        if viewer is not None and n % 10 == 0:
            viewer.sync()
            time.sleep(0.02)


def move_to(q_target, duration, viewer=None):
    q_start = data.ctrl[:7].copy()
    t0 = data.time
    n = 0
    while data.time - t0 < duration and (viewer is None or viewer.is_running()):
        frac = (data.time - t0) / duration
        s = frac * frac * (3 - 2 * frac)
        data.ctrl[:7] = q_start + s * (q_target - q_start)
        mujoco.mj_step(model, data)
        n += 1
        if viewer is not None and n % 10 == 0:
            viewer.sync()
            time.sleep(0.02)
    data.ctrl[:7] = q_target


def set_gripper(value, viewer=None, wait=1.0):
    data.ctrl[7] = value
    run(wait, viewer)


def report(label, box="red_box"):
    #print(f"{label:14s} box={data.body(box).xpos}  hand={data.body('hand').xpos}  "
          #f"fingers={data.qpos[7:9]}")
    pass

reset()



#test to make sure the robotic ARM can move when q for each link is passed
def forward_kinematics():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        run(1.0, viewer)
        report("start")

        move_to(Q_ABOVE_BOX_FK, 2.0, viewer);   report("above box")
        move_to(Q_GRASP_FK, 2.0, viewer);       report("at grasp")
        set_gripper(GRIPPER_CLOSED, viewer); report("closed")
        move_to(Q_ABOVE_BOX_FK, 2.0, viewer);   report("lifted")
        move_to(Q_ABOVE_DROP_FK, 2.5, viewer);  report("above drop")
        move_to(Q_RELEASE_FK, 2.0, viewer);     report("at release")
        set_gripper(GRIPPER_OPEN, viewer);   report("released")
        move_to(Q_ABOVE_DROP_FK, 2.0, viewer);  report("retreated")
        move_to(HOME_Q, 2.0, viewer)

        print("Done. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
            

def execute(Q, box_body, viewer=None):
    run(1.0, viewer)
    report("start", box_body)

    move_to(Q["above_box"], 2.0, viewer);   report("above box", box_body)
    move_to(Q["grasp"], 2.0, viewer);       run(0.5, viewer); report("at grasp", box_body)
    set_gripper(GRIPPER_CLOSED, viewer);    report("closed", box_body)
    move_to(Q["above_box"], 2.0, viewer);   report("lifted", box_body)
    move_to(Q["above_drop"], 2.5, viewer);  report("above drop", box_body)
    move_to(Q["release"], 2.0, viewer);     run(0.5, viewer); report("at release", box_body)
    set_gripper(GRIPPER_OPEN, viewer);      report("released", box_body)
    move_to(Q["above_drop"], 2.0, viewer);  report("retreated", box_body)
    move_to(HOME_Q, 2.0, viewer)


def inverse_kinematics(box_xy, drop_xy, box_body, show_viewer=False):
    """Plan with IK and run the pick-and-place.
    Returns False if IK failed, True if the motion ran.
    NOTE: True does NOT mean the box reached the zone; there is no success check yet."""
    TCP_OFFSET = 0.10       # hand origin is ~10 cm above the fingertip centre
    #drop_xy = (0.5, -0.2)
    box_z = 0.02            # box centre height = where the fingertips should be to grasp
    hover = 0.12            # extra clearance for the "above" poses

    def hand_xyz(xy, tcp_z):
        return np.array([xy[0], xy[1], tcp_z + TCP_OFFSET])

    targets = {
        "above_box":  hand_xyz(box_xy,  box_z + hover),
        "grasp":      hand_xyz(box_xy,  box_z),
        "above_drop": hand_xyz(drop_xy, box_z + hover),
        "release":    hand_xyz(drop_xy, box_z + 0.01),   # tiny margin so the box isn't pressed into the table
    }

    # Top-down orientation, taken once, from a scratch copy so the live sim is untouched
    scratch = mujoco.MjData(model)
    _, down = ik.hand_pose(model, scratch, HOME_Q)

    # Solve in order, seeding each solve with the previous solution
    q = HOME_Q
    Q = {}
    for name, xyz in targets.items():
        q, ok = ik.solve_ik(model, xyz, down, q)
        if not ok:
            print(f"IK failed for '{name}' at {xyz}", flush=True)
            return False
        Q[name] = q

    if show_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            execute(Q, box_body, viewer)
    else:
        execute(Q, box_body)
    return True


def run_batch(n_episodes, seed=0):
    """Run `n_episodes` headless pick-and-place episodes with random layouts and tasks."""
    rng = np.random.default_rng(seed)       # seeded, so a failing episode can be reproduced
    boxes = [color for color, obj_type in OBJECTS.items() if obj_type == "box"]
    zones = [color for color, obj_type in OBJECTS.items() if obj_type == "zone"]

    results = []
    n_done = 0
    n_success = 0
    t_start = time.perf_counter()

    for i in range(n_episodes):
        layout = sample_layout(rng)
        reset(layout["red"], layout["blue"], layout["white"], layout["green"], layout["pink"])

        random_box = rng.choice(boxes)
        random_zone = rng.choice(zones)
        instruction = f"put the {random_box} box in the {random_zone} zone"

        completed = inverse_kinematics(layout[random_box], layout[random_zone], random_box + "_box")
        n_done += bool(completed)
        success = is_success(random_box+"_box",random_zone+"_zone")
        success = completed and is_success(random_box + "_box", random_zone + "_zone")
        n_success += bool(success)
        
        results.append({"episode": i, "instruction": instruction, "completed": bool(completed), "success": success})

        print(f"{i:3d}  {instruction:40s}  {'done' if success else 'fail'}  "
              f"{time.perf_counter() - t_start:6.1f}s", flush=True)

    print(f"{n_success}/{n_episodes} succeeded in {time.perf_counter() - t_start:.1f}s"
          f"in {time.perf_counter() - t_start:.1f}s", flush=True)
    return results

def is_success(box_body, zone_body, tol = 0.04):
    box_x, box_y, _ = data.body(box_body).xpos
    zone_x, zone_y, _ = data.body(zone_body).xpos

    distance =  np.sqrt(np.square(box_x-zone_x) + np.square(box_y-zone_y))
    if distance < tol:
        return True
    return False

if __name__ == "__main__":
    '''
    layout = sample_layout(rng)
    reset(layout["red"], layout["blue"], layout["white"], layout["green"], layout["pink"])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        run(1.0, viewer)                    # let everything settle
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
    
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    with open("behavioural_cloning/results/run1_greytable_100ep/eval_episodes.json") as f:
        eps = json.load(f)["episodes"]
    xy = np.array([e["box_xy"] for e in eps])
    ok = np.array([e["success"] for e in eps])

    edges_x = np.linspace(0.30, 0.70, 5)
    edges_y = np.linspace(-0.30, 0.30, 5)
    rate = np.full((4, 4), np.nan)           # rows: x bins, columns: y bins
    count = np.zeros((4, 4), int)
    for i in range(4):
        for j in range(4):
            m = ((xy[:, 0] >= edges_x[i]) & (xy[:, 0] < edges_x[i + 1]) &
                (xy[:, 1] >= edges_y[j]) & (xy[:, 1] < edges_y[j + 1]))
            count[i, j] = m.sum()
            if m.sum():
                rate[i, j] = ok[m].mean()

    plt.imshow(rate.T, origin="lower", extent=[0.3, 0.7, -0.3, 0.3], vmin=0, vmax=1, cmap="viridis")
    for i in range(4):
        for j in range(4):
            plt.text((edges_x[i] + edges_x[i + 1]) / 2, (edges_y[j] + edges_y[j + 1]) / 2,
                    f"{rate[i, j]:.0%}\n(n={count[i, j]})", ha="center", va="center", color="w")
    plt.colorbar(label="success rate")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Success rate by start position of the target box")
    plt.savefig("behavioural_cloning/results/run1_greytable_100ep/success_heatmap.png", dpi=150)
    plt.show()
    
    from env import OBJECTS

    val = np.load("behavioural_cloning/data/val.npz")
    names = list(OBJECTS)
    box_xy = np.array([val["layout"][i, names.index(str(val["box_color"][i]))]
                    for i in range(len(val["box_color"]))])
    errs = np.load("behavioural_cloning/results/run1_greytable_100ep/offline_step_errors.npy")

    region = (box_xy[:, 0] < 0.45) & (box_xy[:, 1] > 0.1)      # the top-left corner of the scatter
    for step in [0, 1]:
        print(f"step {step} error: corner {errs[region, step].mean():.3f} | elsewhere {errs[~region, step].mean():.3f}  (n={region.sum()})")
    '''
    import numpy as np
    from env import PickPlaceEnv, sample_layout
    from gif_recorder import Recorder

    env = PickPlaceEnv()
    env.reset(sample_layout(np.random.default_rng(0)))

    rec = Recorder(env, every=20, fps=25)
    env.on_step = rec
    env.run(3.0)            # 3 seconds of idle physics
    env.on_step = None
    rec.save()