"""Scripted extraction of one weed: reach in, take the stalk, twist, pull.

The gripper comes in on the stalk from the side rather than down its own axis, closes
across it, turns about the world vertical and lifts. Turning about the vertical is what
lets this work at a force the arm can comfortably hold: a root fails on combined tension
and torsion, so the twist carries part of the load the pull would otherwise need.

The motion is Cartesian throughout. Every control tick a waypoint is placed along the
current phase and handed to the IK, which returns joint angles for the position servos,
so the arm is servoed toward the stalk rather than played back from a recording, and
the collision limits apply the whole way.
"""

import numpy as np

from weed_extraction.ik import ArmIK, pose

# How far up the stalk to take it. Lower and the fingertip pads wedge into the bed;
# this is inside the rigid lower 40 mm, so the whole plant turns with the gripper.
# The arm settles about 2 mm below its commanded tool point, so 14 mm left the pads
# touching the bed at every weed but shepherd_1; 16 mm clears it at all eight.
GRASP_HEIGHT = 0.016

STANDOFF = 0.14         # metres back from the stalk to line up before reaching in
HEADROOM = 0.07         # metres above the plant's own tip to line up, clear of its leaves
LIFT = 0.25             # metres to raise the weed once it is held
# How far to turn the stalk. Small, because the grasp is rigid: the arm turns the plant
# against a stiff root and the torque climbs with the angle, so anything past about 15
# shears the root on the twist alone. At 10 the twist carries two thirds of the root's
# capacity and the pull has to find the rest, which is what twisting a weed is for.
TWIST = np.radians(10)

OPEN = 255.0   # gripper command for full travel, 80 mm between the fingertips
# What to close to. Narrower than the stem so the jaws squeeze it, but a real width and
# not zero: told to shut completely they drive through a 3 mm stalk and close on each
# other, and the plant ends up sitting inside the shut fingers rather than held by them.
SHUT = 6.0     # about 1.9 mm, against a 3.2 mm stem

# How far off vertical the gripper comes in. At 0 it reaches straight down the stalk's
# own axis; at 90 it is level with the bed. 40 is the most side-on the arm can actually
# get to: the gripper itself fits at any angle up to 75, but past 40 there is no path
# down to the stalk that keeps the fingers off the bed, and the solve stalls 70-110 mm
# short -- which looks like a grasp, closes on nothing, and then sinks toward the target
# during the twist.
TILT = np.radians(40)


def smoothstep(t):
    """Ease in and out, so no phase starts or ends with a velocity step."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def between(start, end, t):
    """Straight line from `start` to `end`, eased."""
    return np.asarray(start) + (np.asarray(end) - np.asarray(start)) * smoothstep(t)


def stem_centre(model, data, weed, height):
    """A point on the stalk, `height` above where the weed is planted.

    The stalk runs straight up the weed body's own z axis. Anything derived from the
    collision pieces lands out in the foliage instead: they are leaves as much as stem,
    and spread far enough to one side that their middle misses the stalk by millimetres
    -- which on a 3 mm stem is the difference between a grasp and closing on air.
    """
    body = model.body(weed).id
    return data.xpos[body] + data.xmat[body].reshape(3, 3)[:, 2] * height


class Uprooter:
    """Drives `env` through one extraction and reports what happened.

    Call `step()` once per control tick; it returns the phase it is in, or None when
    the sequence has finished.
    """

    def __init__(self, env, weed="shepherd_1", ik=None):
        self.env, self.weed = env, weed
        self.ik = ik or ArmIK()
        self.root = env.roots.of_body[env.model.body(weed).id]
        self.dt = env.n_substeps * env.model.opt.timestep
        self.grip = stem_centre(env.model, env.data, weed, GRASP_HEIGHT)
        # The rail runs along y = 0, so the gripper always reaches out across the
        # walkway at whichever bed the weed is planted in, tilted back from vertical.
        sideways = np.array([0.0, np.sign(self.grip[1]), 0.0])
        self.approach = np.sin(TILT) * sideways - np.cos(TILT) * np.array([0.0, 0.0, 1.0])
        # Line up clear of the plant's own tip and back along the approach, so the run-in
        # never drags the wrist sideways through the foliage or the bed.
        tip = env.data.xpos[env.model.body(f"{weed}_s3").id][2]
        standoff = self.grip - self.approach * STANDOFF
        standoff[2] = tip + HEADROOM
        lifted = self.grip + [0, 0, LIFT]
        self.phases = [
            ("reach", 3.0, lambda t: (standoff, OPEN, 0.0)),
            ("settle", 1.5, lambda t: (standoff, OPEN, 0.0)),
            ("close in", 4.0, lambda t: (between(standoff, self.grip, t), OPEN, 0.0)),
            ("grip", 1.0, lambda t: (self.grip, between(OPEN, SHUT, t), 0.0)),
            ("twist", 2.0, lambda t: (self.grip, SHUT, TWIST * smoothstep(t))),
            ("lift", 3.0, lambda t: (between(self.grip, lifted, t), SHUT, TWIST)),
            ("retreat", 3.0, lambda t: (between(lifted, standoff + [0, 0, LIFT], t),
                                        SHUT if t < 0.8 else OPEN, TWIST)),
        ]
        self.phase, self.elapsed = 0, 0.0
        self.peak_load = self.peak_pull = self.peak_twist = 0.0
        self.uprooted = None
        self.missed = None   # set if the jaws close with the stalk not between them
        # The base is commanded once and then left alone; its servo takes it there
        # during the reach. Re-deriving a target every tick is what made it pace back
        # and forth, because the phase clock restarts at every phase.
        env.data.ctrl[env.model.actuator("base_x").id] = self.grip[0]

    @property
    def captured(self):
        """True when the stalk is between the jaws, where closing them would catch it.

        Worth asking before closing rather than after: the jaws happily shut on air
        beside the plant and nothing about the gripper command says otherwise.
        """
        model, data = self.env.model, self.env.data
        site = data.site_xpos[model.site("grasp_site").id]
        across = data.site_xmat[model.site("grasp_site").id].reshape(3, 3)[:, 1]
        offset = abs(float(np.dot(self.grip - site, across)))
        half = data.qpos[model.joint("finger_joint1").qposadr[0]]
        return offset < half

    def step(self):
        if self.phase >= len(self.phases):
            return None
        name, duration, waypoint = self.phases[self.phase]
        target, grip, twist = waypoint(self.elapsed / duration)
        angles, _ = self.ik.solve(self.env.data.qpos, pose(target, self.approach, twist))
        self.env.data.ctrl[:7] = angles
        self.env.data.ctrl[7] = grip

        # Take hold the moment the fingertips are genuinely on the stalk, and let go
        # when the jaws are asked to open. Friction alone cannot hold a weed against
        # its own root -- see env.Grasp.
        grasp = self.env.grasp
        if grip <= SHUT and grasp.held is None and grasp.closed(self.env.data, self.weed):
            grasp.take(self.env.data, self.weed)
        elif grip >= OPEN:
            grasp.let_go(self.env.data)

        self._measure()
        self.elapsed += self.dt
        if self.elapsed >= duration:
            self.phase, self.elapsed = self.phase + 1, 0.0
        return name

    def _measure(self):
        data, roots = self.env.data, self.env.roots
        if self.phases[self.phase][0] == "grip" and self.missed is None:
            self.missed = not self.captured
        if data.eq_active[self.root]:
            self.peak_pull = max(self.peak_pull, roots.axial_force(data, self.root))
            self.peak_twist = max(self.peak_twist, roots.torsion(data, self.root))
            self.peak_load = max(self.peak_load, roots.load(data, self.root))
        elif self.uprooted is None:
            self.uprooted = data.time

    @property
    def holding(self):
        """True while the weed is actually held, not merely touched."""
        return self.env.grasp.held == self.weed
