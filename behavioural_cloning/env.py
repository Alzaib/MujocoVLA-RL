"""env.py: the simulated pick-and-place world (model, data, reset, stepping, success check)."""
import time
from pathlib import Path

import mujoco
import numpy as np

# custom scene with panda robot, table, cubes and drop zones
SCENE = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "franka_emika_panda" / "panda_pick_scene.xml"

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0

BOX_Z = 0.02        # centre height of a cube resting on the table
ZONE_Z = 0.0005     # centre height of a drop zone (a thin slab on the table)

# table spans x -0.2..0.8, y -0.4..0.4; we sample inside it with a margin
X_RANGE = (0.30, 0.70)
Y_RANGE = (-0.30, 0.30)

OBJECTS = {"red": "box", "blue": "box", "white": "box", "green": "zone", "pink": "zone"}
RADIUS = {"box": 0.05, "zone": 0.06}      # keep-out radius per object type
BOX_COLORS = [c for c, kind in OBJECTS.items() if kind == "box"]
ZONE_COLORS = [c for c, kind in OBJECTS.items() if kind == "zone"]


def sample_layout(rng, max_tries=1000):
    """Random (x, y) for every object, with no two keep-out circles overlapping."""
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


class PickPlaceEnv:
    def __init__(self, image_size=128):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.on_step = None

        # look the indices up once, instead of by name on every reset
        self._box_qpos_adr = {
            c: int(self.model.joint(f"{c}_box_joint").qposadr[0]) for c in BOX_COLORS}
        self._zone_mocap_id = {
            c: int(self.model.body_mocapid[self.model.body(f"{c}_zone").id]) for c in ZONE_COLORS}
        self.image_size = image_size
        self._renderer = None

    def reset(self, layout):
        """Put the arm at home and every object at layout[color] = (x, y)."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:7] = HOME_Q
        d.qpos[7:9] = 0.04              # fingers open
        d.ctrl[:7] = HOME_Q             # targets = state, so the arm holds still
        d.ctrl[7] = GRIPPER_OPEN

        for color, adr in self._box_qpos_adr.items():
            x, y = layout[color]
            d.qpos[adr:adr + 7] = [x, y, BOX_Z, 1, 0, 0, 0]
        for color, mocap_id in self._zone_mocap_id.items():
            x, y = layout[color]
            d.mocap_pos[mocap_id] = [x, y, ZONE_Z]

        mujoco.mj_forward(m, d)         # refresh derived quantities (xpos etc.)

    def render(self, camera="agentview"):
        """RGB image from a named camera, shape (H, W, 3), dtype uint8."""
        if self._renderer is None:           # created on first use, so headless batches pay nothing
            self._renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    # ---- stepping ----------------------------------------------------------
    def run(self, duration, viewer=None):
        """Let the physics run for `duration` seconds of simulated time."""
        t0 = self.data.time
        n = 0
        while self.data.time - t0 < duration and (viewer is None or viewer.is_running()):
            mujoco.mj_step(self.model, self.data)
            if self.on_step:
                self.on_step()
            n += 1
            if viewer is not None and n % 10 == 0:
                viewer.sync()
                time.sleep(0.02)

    def move_to(self, q_target, duration, viewer=None):
        """Slide the arm's joint targets from where they are now to q_target."""
        q_start = self.data.ctrl[:7].copy()
        t0 = self.data.time
        n = 0
        while self.data.time - t0 < duration and (viewer is None or viewer.is_running()):
            frac = (self.data.time - t0) / duration
            s = frac * frac * (3 - 2 * frac)            # smoothstep
            self.data.ctrl[:7] = q_start + s * (q_target - q_start)
            mujoco.mj_step(self.model, self.data)
            if self.on_step:
                self.on_step()
            n += 1
            if viewer is not None and n % 10 == 0:
                viewer.sync()
                time.sleep(0.02)
        self.data.ctrl[:7] = q_target

    def set_gripper(self, value, viewer=None, wait=1.0):
        self.data.ctrl[7] = value
        self.run(wait, viewer)

    # ---- reading the world -------------------------------------------------
    def box_pos(self, color):
        return self.data.body(f"{color}_box").xpos.copy()

    def zone_pos(self, color):
        return self.data.body(f"{color}_zone").xpos.copy()

    def distance(self, box_color, zone_color):
        """Distance in the table plane between a box centre and a zone centre."""
        return float(np.linalg.norm(self.box_pos(box_color)[:2] - self.zone_pos(zone_color)[:2]))

    def is_success(self, box_color, zone_color, tol=0.04):
        """Box centre within `tol` of the zone centre, and resting on the table."""
        return bool(self.distance(box_color, zone_color) < tol
                    and self.box_pos(box_color)[2] < 0.03)