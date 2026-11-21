"""ik.py: simple damped-least-squares IK for the Panda hand."""
from pathlib import Path
import mujoco
import numpy as np

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])


# Step 1: forward kinematics. Joint angles in, hand pose out.
def hand_pose(model, data, q):
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)          # computes body poses from qpos
    hand = model.body("hand").id
    pos = data.xpos[hand].copy()            # 3 numbers
    rot = data.xmat[hand].reshape(3, 3).copy()   # 3x3 rotation matrix
    return pos, rot


# Step 2: orientation error as a rotation vector (not an Euler-angle difference).
def rotation_error(rot, rot_target):
    return 0.5 * sum(np.cross(rot[:, i], rot_target[:, i]) for i in range(3))


# Step 3: the IK loop.
def solve_ik(model, target_pos, target_rot, q_start, iters=300, damping=0.05, step=0.5):
    data = MjData_copy = mujoco.MjData(model)       # scratch data, never touches the live sim
    hand = model.body("hand").id
    lo, hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    q = np.array(q_start, dtype=float)

    for _ in range(iters):
        pos, rot = hand_pose(model, data, q)

        # error = how far the hand is from the target (position and orientation)
        err = np.concatenate([target_pos - pos, rotation_error(rot, target_rot)])
        if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-3:
            return q, True

        # Jacobian: 6 x 7 (rows: x y z and the 3 rotations, columns: the 7 joints)
        mujoco.mj_jacBody(model, data, jacp, jacr, hand)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])

        # damped least squares: dq = J^T (J J^T + lambda^2 I)^-1 * err
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)

        # Euler update, then respect the joint limits
        q = np.clip(q + step * dq, lo, hi)

    return q, False


# Step 4: test it.
if __name__ == "__main__":
    scene = (Path(__file__).resolve().parent.parent / "mujoco_menagerie"
             / "franka_emika_panda" / "panda_pick_scene.xml")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    np.set_printoptions(precision=4, suppress=True)

    _, down = hand_pose(model, data, HOME_Q)        # home orientation = gripper pointing down
    target = np.array([0.5, 0.2, 0.12])             # hand position, not fingertip position

    q, ok = solve_ik(model, target, down, HOME_Q)
    pos, _ = hand_pose(model, data, q)
    print("success:", ok)
    print("q:", q)
    print("hand reached:", pos, " error (m):", np.linalg.norm(target - pos))