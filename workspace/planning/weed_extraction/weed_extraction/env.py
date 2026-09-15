"""Gymnasium environment for the mobile Panda in the weed field."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

SCENE = (
    Path(__file__).resolve().parents[3]
    / "application"
    / "field_description"
    / "scene_mobile_field.xml"
)

ROOT_BREAK_FORCE = 15.0  # newtons; a small annual weed pulls out at roughly 5-30 N
# Newton-metres of twist that would shear the root on its own. A root fails in torsion
# as well as in tension -- twisting is how you break the fine roots without having to
# out-pull the whole taproot -- so the two share one failure criterion (see Roots.load).
# The same soil friction resists both, acting at the root's own radius, so the torque
# it can take is ROOT_BREAK_FORCE times that radius: 15 N across a 2.7 mm taproot.
# Twisting is cheap, which is exactly why gardeners do it.
ROOT_BREAK_TORQUE = 0.07


def reset_to_home(model, data):
    """Robot to its home keyframe, weeds to their planted pose.

    The keyframe describes the robot only, and MuJoCo pads it with zeros for the
    weeds' free joints, which would teleport them all to the origin.
    """
    key = model.key("home").id
    rows = model.jnt_qposadr[model.jnt_type != mujoco.mjtJoint.mjJNT_FREE]
    mujoco.mj_resetData(model, data)
    data.qpos[rows] = model.key_qpos[key][rows]
    data.ctrl[:] = model.key_ctrl[key]
    mujoco.mj_forward(model, data)


class Roots:
    """The weeds' breakable root welds.

    MuJoCo equalities have no break force, so a weed is released from here instead.
    Three rules make that behave like a real root:

    Only the pull along the plant's own axis counts. A root resists being drawn out
    of the soil; a sideways load bends the stalk and, in a real plant, would lodge or
    snap it rather than extract it. Taking the whole force magnitude would let a purely
    horizontal shove uproot the plant.

    The load has to be sustained, not instantaneous. A root fails under a pull that is
    held, and this also ignores the single-step force spikes that a stiff contact or a
    stiff drag spring can throw into the solver.

    Twist counts alongside pull. A root shears in torsion as well as in tension, which
    is why twisting a weed loose takes less force than hauling it straight out; the two
    are combined in `load`.
    """

    SUSTAIN = 0.02  # seconds the pull must be held before the root gives way

    def __init__(self, model):
        self.model = model
        self.eqs = [i for i in range(model.neq) if (model.eq(i).name or "").endswith("_root")]
        self.base = {i: model.body(model.eq(i).name.removesuffix("_root")).id for i in self.eqs}
        # Grabbing a plant anywhere pulls on the same root, so map its whole subtree.
        self.of_body = {b: i for i in self.eqs for b in range(model.nbody)
                        if model.body_rootid[b] == self.base[i]}
        # The direction the plant was planted in, taken once from the model. It must
        # not follow the body: a plant that has been bent over would otherwise count
        # a sideways pull as axial, and could be uprooted from any direction.
        self.up = {}
        for eq in self.eqs:
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.body_quat[self.base[eq]])
            self.up[eq] = rot.reshape(3, 3)[:, 2]
        self._overloaded_since = {}

    def forget(self):
        """Drop the overload timers; call when the simulation is reset."""
        self._overloaded_since.clear()

    def holds(self, body_id, data):
        """The root still anchoring this body, or None if it is loose or not a weed."""
        eq = self.of_body.get(body_id)
        return eq if eq is not None and data.eq_active[eq] else None

    def _rows(self, data, eq):
        """The weld's six constraint rows: three of force, then three of torque."""
        n = data.nefc
        is_eq = data.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        rows = np.flatnonzero(is_eq & (data.efc_id[:n] == eq))
        return rows if len(rows) == 6 else None

    def axial_force(self, data, eq):
        """Newtons drawing the weed out of the soil along its own axis.

        Sideways load is excluded, and so is downward: pressing a plant into the
        ground is not a way to uproot it.
        """
        rows = self._rows(data, eq)
        if rows is None:
            return 0.0
        # efc_force is the constraint's reaction, so it opposes the pull: an upward
        # pull shows as a downward reaction, hence the sign.
        return max(0.0, -float(np.dot(data.efc_force[rows[:3]], self.up[eq])))

    def torsion(self, data, eq):
        """Newton-metres twisting the weed about its own axis, either way round."""
        rows = self._rows(data, eq)
        if rows is None:
            return 0.0
        return abs(float(np.dot(data.efc_force[rows[3:]], self.up[eq])))

    def load(self, data, eq):
        """How close the root is to failing, as a fraction of its capacity.

        Tension and torsion share one linear interaction criterion, so a twist that
        is not enough to shear the root on its own still lowers the pull needed to
        draw it out -- which is why twisting a weed loose works.
        """
        return (self.axial_force(data, eq) / ROOT_BREAK_FORCE
                + self.torsion(data, eq) / ROOT_BREAK_TORQUE)

    def snap(self, data, threshold=1.0):
        """Release weeds loaded past `threshold` for long enough, and name them."""
        gave_way = []
        for eq in self.eqs:
            if not data.eq_active[eq]:
                continue
            if self.load(data, eq) <= threshold:
                self._overloaded_since.pop(eq, None)
                continue
            since = self._overloaded_since.setdefault(eq, data.time)
            if since > data.time:  # the clock was rewound by a reset
                since = self._overloaded_since[eq] = data.time
            if data.time - since >= self.SUSTAIN:
                data.eq_active[eq] = 0
                gave_way.append(self.model.eq(eq).name.removesuffix("_root"))
        return gave_way


class Grasp:
    """The weld that holds a grasped weed to the hand.

    MuJoCo grippers hold things by friction, and this one cannot: the Panda's gripper
    servo is about 0.1 N per mm, so on a 3 mm stalk it squeezes ~1.3 N and holds ~2 N,
    against a root that takes 15 N. Left to friction the jaws simply close through the
    stem and shut on each other, and what reads as grip is the solver pushing
    overlapping shapes apart. So the hold is made explicit, the same way the root is.

    What keeps that honest is `closed`: a geometric test that both fingertips are
    actually on the plant, not a look at what the gripper was commanded to do.
    """

    PADS = ("left_finger", "right_finger")
    # How far the second fingertip may still be from the stem when the first is on it.
    # The fingers are linked, so the first pad to land stops both; the stem is welded
    # rigidly at the root and cannot give the fraction of a millimetre a real one would
    # to centre itself, and an arm that settles 0.23 mm off-centre leaves the other pad
    # 0.3-0.7 mm short however long the jaws squeeze.
    REACH = 0.001

    def __init__(self, model):
        self.model = model
        self.welds = {model.eq(i).name.removesuffix("_grasp"): i for i in range(model.neq)
                      if (model.eq(i).name or "").endswith("_grasp")}
        self.hand = model.body("hand").id
        self.held = None
        # Only the pads collide; the finger shafts are visual.
        self.pads = {name: [g for g in range(model.ngeom)
                            if model.geom_bodyid[g] == model.body(name).id
                            and (model.geom_contype[g] or model.geom_conaffinity[g])]
                     for name in self.PADS}
        self.stems = {weed: next(g for g in range(model.ngeom)
                                 if model.geom_bodyid[g] == model.body(weed).id
                                 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE)
                      for weed in self.welds}

    def closed(self, data, weed):
        """True when both fingertips are on the weed, or one is on its stem and the
        other is within REACH of it."""
        model = self.model
        root = model.body(weed).id
        stem = self.stems[weed]
        touching, on_stem = set(), set()
        for contact in data.contact[:data.ncon]:
            bodies = [model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]]
            if any(model.body_rootid[b] == root for b in bodies):
                fingers = {model.body(b).name for b in bodies} & set(self.PADS)
                touching |= fingers
                if stem in (contact.geom1, contact.geom2):
                    on_stem |= fingers
        if len(touching) == len(self.PADS):
            return True
        if len(on_stem) != 1:
            return False
        (other,) = set(self.PADS) - on_stem
        return min(mujoco.mj_geomDistance(model, data, g, stem, 2 * self.REACH, None)
                   for g in self.pads[other]) <= self.REACH

    def take(self, data, weed):
        """Weld the weed to the hand exactly where it is, so nothing jumps."""
        eq = self.welds[weed]
        body = self.model.body(weed).id
        rotation = data.xmat[self.hand].reshape(3, 3)
        inverse = np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[self.hand])
        relative = np.zeros(4)
        mujoco.mju_mulQuat(relative, inverse, data.xquat[body])
        self.model.eq_data[eq, :3] = 0.0
        self.model.eq_data[eq, 3:6] = rotation.T @ (data.xpos[body] - data.xpos[self.hand])
        self.model.eq_data[eq, 6:10] = relative
        data.eq_active[eq] = 1
        self.held = weed

    def let_go(self, data):
        """Drop whatever is held."""
        if self.held is not None:
            data.eq_active[self.welds[self.held]] = 0
            self.held = None


class WeedExtractionEnv(gym.Env):
    """Action: absolute targets for the 9 actuators (7 arm joints, gripper 0-255, base x in m).

    Observation: joint positions and velocities, plus one image per camera. Cameras whose
    name ends in "_depth" give float32 depth in metres; the rest give uint8 RGB.
    Reward is 0; no task is defined yet. info["uprooted"] names the weeds torn loose on
    that step.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        scene=SCENE,
        cameras=("d435i_color",),
        width=640,
        height=360,
        control_hz=20,
        max_steps=500,
        render_mode=None,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.cameras = list(cameras)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.n_substeps = round(1 / (control_hz * self.model.opt.timestep))
        self.roots = Roots(self.model)
        self.grasp = Grasp(self.model)
        self.width, self.height = width, height
        self._renderer = None  # built on demand: no cameras means no GL context

        # The offscreen buffer must be at least as large as the requested images.
        vis = self.model.vis.global_
        vis.offwidth, vis.offheight = max(width, vis.offwidth), max(height, vis.offheight)

        ctrl = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = gym.spaces.Box(ctrl[:, 0], ctrl[:, 1], dtype=np.float32)
        spaces = {
            "qpos": gym.spaces.Box(-np.inf, np.inf, (self.model.nq,), np.float64),
            "qvel": gym.spaces.Box(-np.inf, np.inf, (self.model.nv,), np.float64),
        }
        for cam in self.cameras:
            if cam.endswith("_depth"):
                spaces[cam] = gym.spaces.Box(0, np.inf, (height, width), np.float32)
            else:
                spaces[cam] = gym.spaces.Box(0, 255, (height, width, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(spaces)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_to_home(self.model, self.data)
        self.roots.forget()
        self.grasp.let_go(self.data)
        self._steps = 0
        return self.get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        uprooted = self.roots.snap(self.data)
        self._steps += 1
        return (self.get_obs(), 0.0, False, self._steps >= self.max_steps,
                {"uprooted": uprooted})

    def get_obs(self):
        obs = {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}
        for cam in self.cameras:
            obs[cam] = self._render(cam, depth=cam.endswith("_depth"))
        return obs

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render("rgb_camera")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _render(self, camera, depth=False):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        if depth:
            self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        image = self._renderer.render()
        if depth:
            self._renderer.disable_depth_rendering()
        return image
