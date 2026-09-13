"""Collision-aware inverse kinematics for the arm, on top of mink.

Differential IK: rather than solving for joint angles outright, each call solves a
small QP for the joint velocity that moves `grasp_site` toward the target, then
integrates it. Joint limits and collision avoidance enter the same QP as inequality
constraints, so a solution that would drive the arm into the soil or fold it into
itself is never returned in the first place.
"""

import mujoco
import numpy as np
import mink

from weed_extraction.env import SCENE

ARM = tuple(f"joint{i}" for i in range(1, 8))

# How close things may come, in metres. The solver starts pushing back at APPROACH and
# refuses to close past the clearance. The arm proper stays well off the field, but the
# gripper is the tool: it has to reach the soil surface to take a stalk at its base, so
# it gets a clearance of its own that is just thick enough to keep the pads off the bed.
CLEARANCE = 0.015
GRIPPER_CLEARANCE = 0.003
APPROACH = 0.08

# Panda datasheet joint speeds, rad/s. These are not decoration: collision avoidance is
# a limit on velocity, so it only bites if the solver is stopped from taking a step big
# enough to jump the APPROACH band in one go. Without them a solve to a distant target
# will happily put the elbow through a bed on its way there.
SPEED = {"joint1": 2.175, "joint2": 2.175, "joint3": 2.175, "joint4": 2.175,
         "joint5": 2.610, "joint6": 2.610, "joint7": 2.610}

UP = np.array([0.0, 0.0, 1.0])


def ik_model(scene=SCENE):
    """The scene with the plants removed: robot, ground and beds only.

    The weeds and strawberries are free bodies, and their 96 unactuated DOFs would
    otherwise enter every QP as variables the arm cannot move. Dropping them makes the
    solve 3.5x faster and keeps mink's posture and configuration limits well defined.
    """
    spec = mujoco.MjSpec.from_file(str(scene))
    for key in list(spec.keys):  # sized for the plants, invalid once they go
        spec.delete(key)
    for body in list(spec.worldbody.bodies):
        if body.name != "mobile_base":
            spec.delete(body)
    return spec.compile()


def pose(position, approach, twist=0.0):
    """Target for `grasp_site`: reaching along `approach`, turned `twist` about vertical.

    `approach` is the gripper's z, which is horizontal for a side grasp, so the fingers
    close across an upright stalk rather than around it from above. The twist is about
    the world vertical, not about the gripper's own axis: that is the stalk's own axis,
    and it is the only rotation a root feels as torsion.
    """
    forward = np.asarray(approach, dtype=float)
    forward = forward / np.linalg.norm(forward)
    # `across` is the direction the fingers open along, always horizontal so they close
    # across an upright stalk. Its sign rolls the wrist so the hand's instrument face --
    # and the D435i bolted to it -- looks up rather than down into the soil.
    across = np.cross(forward, UP)
    length = np.linalg.norm(across)
    across = np.array([1.0, 0.0, 0.0]) if length < 1e-9 else across / length
    frame = np.column_stack([np.cross(across, forward), across, forward])
    return mink.SE3.from_rotation_and_translation(
        mink.SO3.from_z_radians(twist) @ mink.SO3.from_matrix(frame),
        np.asarray(position, dtype=float))


class ArmIK:
    """Solves arm poses against a plant-free copy of the scene.

    The base and the fingers are frozen: the base is driven to the weed and stopped
    before the arm moves, and letting IK slide a 6 m rail to satisfy a 0.8 m reach
    would hide every genuinely unreachable target.
    """

    def __init__(self, scene=SCENE):
        self.model = ik_model(scene)
        self.configuration = mink.Configuration(self.model)
        self.task = mink.FrameTask("grasp_site", "site", position_cost=1.0,
                                   orientation_cost=0.5, lm_damping=1.0)
        # Weak enough not to fight the frame task, which otherwise stalls short of
        # the target: at cost 1e-2 the solve settles 0.3 mm out and never closes.
        self.posture = mink.PostureTask(self.model, cost=1e-4)
        self.limits = [mink.ConfigurationLimit(self.model),
                       mink.VelocityLimit(self.model, SPEED), *self._avoid()]
        frozen = [j for j in range(self.model.njnt)
                  if self.model.joint(j).name not in ARM]
        self.frozen = mink.DofFreezingTask(
            self.model, dof_indices=[self.model.jnt_dofadr[j] for j in frozen])
        self.arm_qpos = [self.model.jnt_qposadr[self.model.joint(n).id] for n in ARM]

    def _avoid(self):
        """What must not touch what: the arm off the field, the wrist off the shoulder.

        The gripper is split out of the arm because it is the part that has to get
        close to the soil; everything behind the wrist is kept a full CLEARANCE away.
        """
        def geoms(*bodies):
            return [g for b in bodies
                    for g in mink.get_body_geom_ids(self.model, self.model.body(b).id)
                    if self.model.geom_contype[g] or self.model.geom_conaffinity[g]]

        links = geoms(*(f"link{i}" for i in range(1, 8)))
        gripper = geoms("hand", "d435i", "left_finger", "right_finger")
        shoulder = geoms("link0", "link1", "link2")
        # base_cart is deliberately not here. It is a visual box drawn under link0, so
        # link1 already touches it at zero distance -- asking for any clearance against
        # it makes the QP infeasible before the arm has moved. The arm cannot reach
        # under its own base anyway.
        field = [self.model.geom(n).id for n in ("path", "bed1", "bed2")]
        limit = lambda pairs, clearance: mink.CollisionAvoidanceLimit(
            self.model, geom_pairs=pairs, minimum_distance_from_collisions=clearance,
            collision_detection_distance=APPROACH)
        return [limit([(links, field), (geoms("link5", "link6", "link7"), shoulder)],
                      CLEARANCE),
                limit([(gripper, field), (gripper, shoulder)], GRIPPER_CLEARANCE)]

    def _clamped(self, qpos):
        """The robot's joint angles, pulled back inside their limits.

        The live simulation overshoots a limit by a few microradians -- the gripper's
        tendon does it every time it closes -- and a frozen joint that sits outside its
        range asks the QP to both move it back and hold it still, which has no solution.
        """
        q = np.asarray(qpos)[: self.model.nq].copy()
        limited = self.model.jnt_limited.astype(bool)
        rows = self.model.jnt_qposadr[limited]
        q[rows] = np.clip(q[rows], self.model.jnt_range[limited, 0],
                          self.model.jnt_range[limited, 1])
        return q

    def solve(self, qpos, target, dt=0.02, iterations=200, tolerance=1e-4):
        """Arm joint angles that put `grasp_site` on `target`, starting from `qpos`.

        Returns the seven angles and the residual position error in metres. A target
        the arm cannot reach, or can only reach through the soil, comes back with the
        error it settled at rather than an exception -- the caller decides what to do
        about a pose it cannot make.
        """
        self.configuration.update(self._clamped(qpos))
        self.posture.set_target(self.configuration.q.copy())
        self.task.set_target(target)
        for _ in range(iterations):
            velocity = mink.solve_ik(
                self.configuration, [self.task, self.posture], dt, "daqp",
                damping=1e-3, limits=self.limits, constraints=[self.frozen])
            self.configuration.integrate_inplace(velocity, dt)
            error = self.task.compute_error(self.configuration)
            if np.linalg.norm(error[:3]) < tolerance:
                break
        return self.configuration.q[self.arm_qpos].copy(), np.linalg.norm(error[:3])

    def site_position(self, qpos):
        """Where `grasp_site` sits for a given configuration."""
        self.configuration.update(self._clamped(qpos))
        return self.configuration.get_transform_frame_to_world(
            "grasp_site", "site").translation()
