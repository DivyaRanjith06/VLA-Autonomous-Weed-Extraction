import mujoco
import numpy as np
import pytest

from weed_extraction.env import ROOT_BREAK_FORCE, SCENE, Roots, reset_to_home


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_path(str(SCENE))


def home_data(model):
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    return data


def test_compiles(model):
    assert model.nu == 9
    cameras = [model.camera(i).name for i in range(model.ncam)]
    assert cameras == ["rgb_camera", "d435i_color", "d435i_depth"]


def test_base_reaches_target(model):
    data = home_data(model)
    data.ctrl[model.actuator("base_x").id] = 2.0
    mujoco.mj_step(model, data, nstep=6000)
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert abs(data.qpos[model.joint("base_slide_x").qposadr[0]] - 2.0) < 0.01


def test_d435i_looks_down_at_home(model):
    data = home_data(model)
    for cam in ("d435i_color", "d435i_depth"):
        view_dir = -data.cam_xmat[model.camera(cam).id].reshape(3, 3)[:, 2]
        assert view_dir[2] < -0.99


def plant_geoms(model, weed):
    """Every geom of a weed, across its rigid base and its three stalk segments."""
    out = []
    for name in (weed, f"{weed}_s1", f"{weed}_s2", f"{weed}_s3"):
        b = model.body(name)
        out += [model.geom(i) for i in range(b.geomadr[0], b.geomadr[0] + b.geomnum[0])]
    return out


def test_weed_collides_with_its_decomposition_not_its_hull(model):
    """The visual mesh never collides; MuJoCo would replace it with a convex hull
    20-70 mm wide where the stem is 1-2.5 mm. The convex pieces collide instead."""
    for weed, pieces in (("sowthistle_1", 40), ("shepherd_1", 20)):
        geoms = plant_geoms(model, weed)
        visual = [g for g in geoms if g.contype[0] == 0 and g.conaffinity[0] == 0]
        collision = [g for g in geoms if g.contype[0] != 0]
        assert len(visual) == 4, "one visual band per segment"
        assert len(collision) == pieces
        for g in collision:
            assert model.geom_type[g.id] == mujoco.mjtGeom.mjGEOM_MESH
            assert model.geom_condim[g.id] == 4  # torsional friction: no spinning free
            assert model.geom_group[g.id] == 3   # hidden in the viewer by convention
        assert max(model.geom_rbound[g.id] for g in collision) < 0.09


def test_base_is_rigid_and_the_stalk_above_it_bends(model):
    """The gripper needs something solid to hold, so only the stalk above 4 cm hinges."""
    for weed in ("sowthistle_1", "shepherd_1"):
        base = model.body(weed)
        assert base.jntnum[0] == 1
        assert model.jnt_type[base.jntadr[0]] == mujoco.mjtJoint.mjJNT_FREE
        for s in range(1, 4):
            seg = model.body(f"{weed}_s{s}")
            assert seg.jntnum[0] == 2  # bends about x and y
            for j in range(seg.jntadr[0], seg.jntadr[0] + 2):
                assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                assert model.jnt_stiffness[j] > 0  # springs back upright


def test_stalk_stands_up_then_bends_and_recovers(model):
    data = home_data(model)
    tip = model.body("sowthistle_1_s3").id
    rest = data.xpos[tip].copy()
    mujoco.mj_step(model, data, nstep=3000)
    assert np.linalg.norm(data.xpos[tip] - rest) < 0.002  # barely droops under gravity

    data.xfrc_applied[tip, 0] = 0.15
    mujoco.mj_step(model, data, nstep=800)
    assert np.linalg.norm(data.xpos[tip] - rest) > 0.02  # a light push moves it plenty

    data.xfrc_applied[:] = 0
    mujoco.mj_step(model, data, nstep=3000)
    assert np.linalg.norm(data.xpos[tip] - rest) < 0.002  # and it springs back


def test_weeds_stay_rooted_when_undisturbed(model):
    data = home_data(model)
    roots = Roots(model)
    assert len(roots.eqs) == 8
    planted = {w: data.body(w).xpos.copy() for w in ("sowthistle_1", "shepherd_3")}
    for _ in range(3000):
        mujoco.mj_step(model, data)
        assert not roots.snap(data)
    for weed, pos in planted.items():
        assert np.linalg.norm(data.body(weed).xpos - pos) < 0.002


def uproot(model, data, roots, body, force, steps=2000, ramp=300):
    """Lean on one body and report which roots gave way.

    The force ramps in: slamming newtons onto a 6 g plant in a single step is a
    solver transient, not a pull, and it would swamp what the test is measuring.
    """
    gave_way = []
    for i in range(steps):
        if not gave_way:  # let go once it comes loose, as any real pull would
            data.xfrc_applied[model.body(body).id, :3] = np.array(force) * min(1.0, i / ramp)
        else:
            data.xfrc_applied[:] = 0
        mujoco.mj_step(model, data)
        gave_way += roots.snap(data)
    data.xfrc_applied[:] = 0
    return gave_way


def test_weed_uproots_under_a_firm_upward_pull(model):
    data = home_data(model)
    roots = Roots(model)
    assert uproot(model, data, roots, "sowthistle_1", [0, 0, 25.0]) == ["sowthistle_1"]
    mujoco.mj_step(model, data, nstep=3000)
    assert data.body("sowthistle_1").xpos[2] > 0.10  # loose, resting on the bed
    assert sum(data.eq_active[e] for e in roots.eqs) == 7  # neighbours untouched


def test_only_the_upward_pull_uproots(model):
    """A sideways shove bends the stalk; pressing down cannot uproot at all."""
    roots = Roots(model)
    for force, expected in (([0, 0, 25.0], ["sowthistle_1"]),
                            ([25.0, 0, 0], []),
                            ([0, 0, -25.0], [])):
        data = home_data(model)
        assert uproot(model, data, roots, "sowthistle_1", force) == expected


def test_a_force_spike_alone_does_not_uproot(model):
    """The pull has to be held: one overloaded step is a solver artefact, not a root
    failing. Without this a stiff contact could tear a plant out on its own."""
    data = home_data(model)
    roots = Roots(model)
    body = model.body("sowthistle_1").id
    for _ in range(40):                      # far over the threshold, but only ~1 ms each
        data.xfrc_applied[body, 2] = 200.0
        mujoco.mj_step(model, data)
        data.xfrc_applied[:] = 0
        mujoco.mj_step(model, data, nstep=30)
        assert not roots.snap(data)


def test_an_uprooted_weed_falls_like_a_loose_object(model):
    """It has to drop at gravity and be free to tumble. Damping or armature on the
    free joint buys stability at the cost of both: one value covers all six degrees
    of freedom, and anything big enough to slow a fall dwarfs a plant's rotational
    inertia, so it sinks bolt upright instead of falling."""
    data = home_data(model)
    roots = Roots(model)
    body = model.body("sowthistle_1")
    data.eq_active[roots.of_body[body.id]] = 0            # cut the root, lift, let go
    data.qpos[model.jnt_qposadr[body.jntadr[0]] + 2] += 0.5
    data.qvel[body.dofadr[0] + 4] = 0.6                   # with a little spin, as a hand gives
    mujoco.mj_forward(model, data)

    start = data.body("sowthistle_1").xpos[2]
    for step in range(6000):
        mujoco.mj_step(model, data)
        if start - data.body("sowthistle_1").xpos[2] >= 0.30:
            break
    free_fall = np.sqrt(2 * 0.30 / 9.81)
    assert step * model.opt.timestep < free_fall * 1.1    # not floating down
    tilt = np.arccos(np.clip(data.body("sowthistle_1").xmat.reshape(3, 3)[2, 2], -1, 1))
    assert np.degrees(tilt) > 5                           # and free to topple


def test_crop_is_not_uprootable(model):
    """Strawberries have no joints, so a policy cannot pull one out by accident."""
    for i in range(1, 9):
        assert model.body(f"strawberry_{i}").dofnum[0] == 0
