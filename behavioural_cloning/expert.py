"""expert.py: scripted pick-and-place expert. Box + zone in, joint waypoints out."""
import mujoco
import numpy as np

import ik
from env import HOME_Q, GRIPPER_OPEN, GRIPPER_CLOSED, BOX_Z

TCP_OFFSET = 0.10       # hand origin is ~10 cm above the fingertip centre
HOVER = 0.12            # extra clearance for the "above" poses
RELEASE_MARGIN = 0.01   # so the box isn't pressed into the table on release

# The 7-step sequence: at the start of step k the policy sees an observation and
# must output STEP_TARGETS[k], with the gripper command STEP_GRIPPER[k] after arriving.
STEP_TARGETS = ["above_box", "grasp", "above_box", "above_drop", "release", "above_drop", "home"]
STEP_GRIPPER = [GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_CLOSED, GRIPPER_CLOSED,
                GRIPPER_OPEN, GRIPPER_OPEN, GRIPPER_OPEN]


def plan(env, box_color, zone_color):
    """Joint waypoints for 'put the <box_color> box in the <zone_color> zone'.
    Returns a dict of four (7,) arrays, or None if IK fails. Call after env.reset()."""
    box_xy = env.box_pos(box_color)[:2]
    zone_xy = env.zone_pos(zone_color)[:2]

    def hand_xyz(xy, tcp_z):
        return np.array([xy[0], xy[1], tcp_z + TCP_OFFSET])

    targets = {
        "above_box":  hand_xyz(box_xy,  BOX_Z + HOVER),
        "grasp":      hand_xyz(box_xy,  BOX_Z),
        "above_drop": hand_xyz(zone_xy, BOX_Z + HOVER),
        "release":    hand_xyz(zone_xy, BOX_Z + RELEASE_MARGIN),
    }

    # Top-down orientation, taken once, from a scratch copy so the live sim is untouched
    scratch = mujoco.MjData(env.model)
    _, down = ik.hand_pose(env.model, scratch, HOME_Q)

    # Solve in order, seeding each solve with the previous solution
    q = HOME_Q
    Q = {}
    for name, xyz in targets.items():
        q, ok = ik.solve_ik(env.model, xyz, down, q)
        if not ok:
            return None
        Q[name] = q
    return Q


def step_labels(Q):
    """The 7 targets as arrays: joints (7, 7) and gripper commands (7,)."""
    q = np.stack([HOME_Q if name == "home" else Q[name] for name in STEP_TARGETS])
    return q, np.array(STEP_GRIPPER)


def execute(env, Q, viewer=None, on_step=None):
    """Play the 7 steps. on_step(k) is called at the start of step k, before moving."""
    def start(k):
        if on_step:
            on_step(k)

    env.run(1.0, viewer)                                    # settle at home
    start(0); 
    env.move_to(Q["above_box"], 2.0, viewer)

    start(1); 
    env.move_to(Q["grasp"], 2.0, viewer);  
    env.run(0.5, viewer)
    env.set_gripper(GRIPPER_CLOSED, viewer)
    
    start(2); 
    env.move_to(Q["above_box"], 2.0, viewer)      # lift
    
    start(3); 
    env.move_to(Q["above_drop"], 2.5, viewer)
    
    start(4); 
    env.move_to(Q["release"], 2.0, viewer);  
    env.run(0.5, viewer)
    env.set_gripper(GRIPPER_OPEN, viewer)
    
    start(5); 
    env.move_to(Q["above_drop"], 2.0, viewer)     # retreat
    
    start(6); 
    env.move_to(HOME_Q, 2.0, viewer)