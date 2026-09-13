import mujoco
import numpy as np
import pytest

from weed_extraction.env import (ROOT_BREAK_FORCE, ROOT_BREAK_TORQUE,
                                 WeedExtractionEnv, reset_to_home)
from weed_extraction.ik import ArmIK
from weed_extraction.uproot import GRASP_HEIGHT, Uprooter, stem_centre

WEED = "shepherd_1"


@pytest.fixture(scope="module")
def ik():
    return ArmIK()


@pytest.fixture
def env():
    env = WeedExtractionEnv(cameras=(), control_hz=50)
    env.reset()
    yield env
    env.close()


def hold(env, body, force=np.zeros(3), torque=np.zeros(3), seconds=0.5):
    """Apply a steady wrench to a weed and report the worst load its root saw."""
    root = env.roots.of_body[env.model.body(body).id]
    worst = (0.0, 0.0, 0.0)
    for _ in range(int(seconds / env.model.opt.timestep)):
        env.data.xfrc_applied[:] = 0
        env.data.xfrc_applied[env.model.body(body).id, :3] = force
        env.data.xfrc_applied[env.model.body(body).id, 3:] = torque
        mujoco.mj_step(env.model, env.data)
        if env.data.eq_active[root]:
            worst = max(worst, (env.roots.load(env.data, root),
                                env.roots.axial_force(env.data, root),
                                env.roots.torsion(env.data, root)))
        if env.roots.snap(env.data):
            return worst, True
    return worst, False


def test_a_pure_twist_can_shear_the_root_on_its_own(env):
    """Torsion is a failure mode in its own right, not just a contribution."""
    axis = env.roots.up[env.roots.of_body[env.model.body(WEED).id]]
    (_, pull, twist), gave_way = hold(env, WEED, torque=0.15 * axis, seconds=1.0)
    assert gave_way
    assert twist > ROOT_BREAK_TORQUE
    # Torsion did the work. Some pull comes along for the ride -- turning a plant that
    # is resting on the soil lifts it slightly -- but it is nowhere near enough alone.
    assert pull < 0.5 * ROOT_BREAK_FORCE
    assert twist / ROOT_BREAK_TORQUE > pull / ROOT_BREAK_FORCE


def test_a_twist_too_small_to_shear_it_leaves_it_rooted(env):
    axis = env.roots.up[env.roots.of_body[env.model.body(WEED).id]]
    _, gave_way = hold(env, WEED, torque=0.01 * axis, seconds=1.0)
    assert not gave_way


def test_twisting_while_pulling_uproots_below_the_straight_pull_force(env):
    """The whole point of twisting: it buys you the pull you cannot apply."""
    axis = env.roots.up[env.roots.of_body[env.model.body(WEED).id]]
    half = 0.6 * ROOT_BREAK_FORCE * axis
    (_, pull, _), alone = hold(env, WEED, force=half, seconds=1.0)
    assert not alone and pull < ROOT_BREAK_FORCE

    reset_to_home(env.model, env.data)
    env.roots.forget()
    (_, pull, twist), together = hold(env, WEED, force=half, torque=0.02 * axis,
                                      seconds=1.0)
    assert together
    assert pull < ROOT_BREAK_FORCE and twist < ROOT_BREAK_TORQUE


def test_load_combines_the_two_channels(env):
    root = env.roots.of_body[env.model.body(WEED).id]
    mujoco.mj_forward(env.model, env.data)
    expected = (env.roots.axial_force(env.data, root) / ROOT_BREAK_FORCE
                + env.roots.torsion(env.data, root) / ROOT_BREAK_TORQUE)
    assert env.roots.load(env.data, root) == pytest.approx(expected)


def test_stem_centre_lands_on_the_stalk(env):
    """A few millimetres matters when the stem is 3 mm wide."""
    model, data = env.model, env.data
    centre = stem_centre(model, data, WEED, GRASP_HEIGHT)
    body = model.body(WEED).id
    assert centre[2] == pytest.approx(data.xpos[body][2] + GRASP_HEIGHT)
    stem = next(g for g in range(model.ngeom)
                if model.body_rootid[model.geom_bodyid[g]] == body
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE)
    axis = data.geom_xmat[stem].reshape(3, 3)[:, 2]
    offset = centre - data.geom_xpos[stem]
    assert np.linalg.norm(offset - axis * np.dot(offset, axis)) < model.geom_size[stem, 0]


def test_the_grasp_height_keeps_the_fingertips_out_of_the_soil(env):
    """Below this the pads wedge into the bed and the fingers never close."""
    bed = env.model.geom("bed1")
    top = bed.pos[2] + bed.size[2]
    soil = env.data.xpos[env.model.body(WEED).id][2]
    assert soil + GRASP_HEIGHT - 0.009 > top   # the pads reach 9 mm below the tool point


@pytest.mark.slow
def test_the_arm_grasps_twists_and_pulls_the_weed_out(env, ik):
    runner = Uprooter(env, WEED, ik=ik)
    phases, held_during_lift = [], []
    while True:
        name = runner.step()
        if name is None:
            break
        env.step(env.data.ctrl)
        phases.append(name)
        if name == "lift":
            held_during_lift.append(runner.holding)

    assert runner.uprooted is not None, (
        f"stayed rooted: peak load {runner.peak_load:.2f}, "
        f"pull {runner.peak_pull:.1f} N, twist {runner.peak_twist:.3f} N.m")
    # It has to come out under the gripper, not by being knocked over on the way in.
    assert phases[int(runner.uprooted / runner.dt) - 1] in ("twist", "lift")
    assert any(held_during_lift)
    assert not env.data.eq_active[runner.root]


@pytest.mark.slow
def test_the_approach_does_not_disturb_the_weed(env, ik):
    """Reaching over the plant and coming down through it must not tear it loose."""
    runner = Uprooter(env, WEED, ik=ik)
    while runner.phase < 3:  # reach, hover, descend
        if runner.step() is None:
            break
        env.step(env.data.ctrl)
    assert env.data.eq_active[runner.root]
    assert runner.peak_load < 0.5


@pytest.mark.slow
def test_the_arm_never_drives_into_the_beds_or_the_ground(env, ik):
    """The arm may graze the soil; it must never plough into it.

    Collision avoidance is a limit on joint velocity, applied afresh each control tick.
    The twist asks for a large reconfiguration in one tick, and the solve ends on the
    boundary rather than a full CLEARANCE clear of it, so link5 brushes the bed by
    under a millimetre. That is a tracking artifact, not a path through the soil.
    """
    runner = Uprooter(env, WEED, ik=ik)
    model = env.model
    field = {model.geom(n).id for n in ("path", "bed1", "bed2")}
    arm = {model.body(f"link{i}").id for i in range(1, 8)}
    deepest, culprit = 0.0, None
    while True:
        if runner.step() is None:
            break
        env.step(env.data.ctrl)
        for c in range(env.data.ncon):
            g1, g2 = env.data.contact[c].geom1, env.data.contact[c].geom2
            bodies = {model.geom_bodyid[g1], model.geom_bodyid[g2]}
            if (g1 in field or g2 in field) and bodies & arm:
                if -env.data.contact[c].dist > deepest:
                    deepest = -env.data.contact[c].dist
                    culprit = (round(env.data.time, 2),
                               model.body(next(iter(bodies & arm))).name)
    assert deepest < 0.002, f"arm {1000 * deepest:.1f} mm into the field at {culprit}"
