import mujoco
import numpy as np
import pytest

from weed_extraction.env import (ROOT_BREAK_FORCE, ROOT_BREAK_TORQUE,
                                 WeedExtractionEnv, reset_to_home)
from weed_extraction.ik import ArmIK
from weed_extraction.uproot import Uprooter, stem_centre

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
                                *env.roots.wrench(env.data, root)))
        if env.roots.snap(env.data):
            return worst, True
    return worst, False


def pinch(env, weed, gap, sink=5e-5):
    """Stand `weed`'s stem between the open jaws, pressed `sink` into the left pad and
    `gap` from the right one. Returns the measured left and right pad distances."""
    model, data, grasp = env.model, env.data, env.grasp
    stem = grasp.stems[weed]
    fingers = [model.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")]
    free = model.jnt_qposadr[model.body_jntadr[model.body(weed).id]]

    def distance(side):
        return min(mujoco.mj_geomDistance(model, data, g, stem, 0.05, None)
                   for g in grasp.pads[side])

    def pad(side):
        return data.geom_xpos[grasp.pads[side][0]].copy()

    mujoco.mj_forward(model, data)
    data.qpos[free:free + 3] = (pad("left_finger") + pad("right_finger")) / 2 - model.geom_pos[stem]
    data.qpos[free + 3:free + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    across = pad("left_finger") - pad("right_finger")
    across /= np.linalg.norm(across)
    for _ in range(10):  # the distance query is not linear over tens of mm, so iterate
        data.qpos[fingers] += (gap - sink - distance("left_finger") - distance("right_finger")) / 2
        mujoco.mj_forward(model, data)
        data.qpos[free:free + 3] += across * (distance("left_finger") + sink)
        mujoco.mj_forward(model, data)
    return distance("left_finger"), distance("right_finger")


@pytest.mark.parametrize("gap, closed", [(0.0005, True), (0.002, False)])
def test_one_pad_on_the_stem_is_a_grasp_only_if_the_other_is_close(env, gap, closed):
    """The linked fingers stop when the first pad lands, and a stem welded rigidly at the
    root cannot centre itself, so the second pad can be left a fraction of a mm short."""
    left, right = pinch(env, "sowthistle_2", gap)
    assert left < 0 and right == pytest.approx(gap, abs=1e-4)
    assert env.grasp.closed(env.data, "sowthistle_2") is closed


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
    pull, twist = env.roots.wrench(env.data, root)
    expected = pull / ROOT_BREAK_FORCE + twist / ROOT_BREAK_TORQUE
    assert env.roots.load(env.data, root) == pytest.approx(expected)


def test_stem_centre_lands_on_the_stalk(env):
    """A few millimetres matters when the stem is 3 mm wide."""
    model, data = env.model, env.data
    centre = stem_centre(model, data, WEED, Uprooter.GRASP_HEIGHT)
    body = model.body(WEED).id
    assert centre[2] == pytest.approx(data.xpos[body][2] + Uprooter.GRASP_HEIGHT)
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
    assert soil + Uprooter.GRASP_HEIGHT - 0.009 > top   # the pads reach 9 mm below the tool point


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
def test_the_approach_does_not_tear_out_a_sowthistle(env, ik):
    """Its canopy is wide enough to catch the hand and camera on the way in."""
    runner = Uprooter(env, "sowthistle_1", ik=ik)
    while runner.phase < 3:  # reach, settle, close in
        assert runner.step() is not None
        env.step(env.data.ctrl)
        assert env.data.eq_active[runner.root], f"root tore out at {env.data.time:.2f} s"
    assert runner.phases[runner.phase][0] == "grip"
    assert runner.uprooted is None


@pytest.mark.slow
def test_a_stem_off_centre_in_the_jaws_is_still_held(env, ik):
    """The arm settles 0.23 mm to one side of sowthistle_2, so only one pad reaches it."""
    runner = Uprooter(env, "sowthistle_2", ik=ik)
    held_during_lift = []
    while (name := runner.step()) not in (None, "carry"):
        env.step(env.data.ctrl)
        if name == "lift":
            held_during_lift.append(runner.holding)
    assert held_during_lift and all(held_during_lift)


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


@pytest.mark.parametrize("break_force", [15.0, 17.5, 20.0])
def test_the_root_gives_way_at_the_break_force_set_on_reset(env, break_force):
    env.reset(options={"break_force": break_force})
    axis = env.roots.up[env.roots.of_body[env.model.body(WEED).id]]
    _, below = hold(env, WEED, force=0.9 * break_force * axis, seconds=1.0)
    assert not below
    env.reset(options={"break_force": break_force})
    _, above = hold(env, WEED, force=1.1 * break_force * axis, seconds=1.0)
    assert above


@pytest.mark.parametrize("break_force", [15.0, 17.5, 20.0])
def test_load_uses_the_values_set_on_reset(env, break_force):
    env.reset(options={"break_force": break_force, "break_torque": 0.05})
    root = env.roots.of_body[env.model.body(WEED).id]
    hold(env, WEED, force=[0, 0, 5.0], torque=[0, 0, 0.01], seconds=0.1)
    pull, twist = env.roots.wrench(env.data, root)
    assert pull > 0 and twist > 0
    assert env.roots.load(env.data, root) == pytest.approx(pull / break_force + twist / 0.05)


def test_a_reset_without_options_restores_the_defaults(env):
    env.reset(options={"break_force": 20.0, "break_torque": 0.1})
    env.reset()
    assert env.roots.break_force == ROOT_BREAK_FORCE == 15.0
    assert env.roots.break_torque == ROOT_BREAK_TORQUE == 0.07


def test_a_stronger_root_holds_a_pull_the_nominal_one_gives_way_to(env):
    """Root strength scales one weed's capacity, in pull and twist alike."""
    axis = env.roots.up[env.roots.of_body[env.model.body(WEED).id]]
    pull = 1.2 * ROOT_BREAK_FORCE * axis
    _, nominal = hold(env, WEED, force=pull, seconds=1.0)
    assert nominal
    env.reset(options={"root_strength": {WEED: 20 / 15}})
    _, strong = hold(env, WEED, force=pull, seconds=1.0)
    assert not strong
    env.reset(options={"root_strength": {WEED: 20 / 15}})
    _, stronger_pull = hold(env, WEED, force=1.5 * ROOT_BREAK_FORCE * axis, seconds=1.0)
    assert stronger_pull


def test_root_strength_applies_to_the_named_weed_only(env):
    env.reset(options={"root_strength": {WEED: 1.25}})
    root = env.roots.of_body[env.model.body(WEED).id]
    assert env.roots.strength[root] == 1.25
    assert all(s == 1.0 for eq, s in env.roots.strength.items() if eq != root)
    hold(env, WEED, force=[0, 0, 5.0], torque=[0, 0, 0.01], seconds=0.1)
    pull, twist = env.roots.wrench(env.data, root)
    assert env.roots.load(env.data, root) == pytest.approx(
        (pull / ROOT_BREAK_FORCE + twist / ROOT_BREAK_TORQUE) / 1.25)
    env.reset()
    assert set(env.roots.strength.values()) == {1.0}


def test_root_strength_for_a_weed_that_is_not_there_is_refused(env):
    with pytest.raises(ValueError, match="dandelion_1"):
        env.reset(options={"root_strength": {"dandelion_1": 1.2}})
