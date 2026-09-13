import mujoco
import numpy as np
import pytest

from weed_extraction.env import WeedExtractionEnv
from weed_extraction.ik import ARM, CLEARANCE, ArmIK, ik_model, pose

DOWN = [0.0, 0.0, -1.0]
SIDEWAYS = [0.0, -1.0, 0.0]


@pytest.fixture(scope="module")
def ik():
    return ArmIK()


@pytest.fixture
def env():
    env = WeedExtractionEnv(cameras=())
    env.reset()
    yield env
    env.close()


def test_ik_model_has_the_robot_and_the_field_but_no_plants():
    model = ik_model()
    assert model.nq == model.nv == 10  # the rail, 7 arm joints, 2 fingers
    assert [model.joint(j).name for j in range(model.njnt)][1:8] == list(ARM)
    bodies = [model.body(b).name for b in range(model.nbody)]
    assert not [b for b in bodies if "sowthistle" in b or "shepherd" in b
                or "strawberry" in b]
    for geom in ("path", "bed1", "bed2"):  # the ground is still there to avoid
        assert model.geom(geom).id >= 0


def test_grasp_site_sits_between_the_fingertips(env):
    """IK aims at the tool centre point, not the wrist flange."""
    model, data = env.model, env.data
    site = data.site_xpos[model.site("grasp_site").id]
    pads = [data.geom_xpos[g] for b in ("left_finger", "right_finger")
            for g in range(model.ngeom)
            if model.geom_bodyid[g] == model.body(b).id
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
    assert np.linalg.norm(site - np.mean(pads, axis=0)) < 0.01


def test_reaches_a_pose_above_a_weed(ik, env):
    weed = env.data.xpos[env.model.body("shepherd_1").id]
    q, error = ik.solve(env.data.qpos, pose(weed + [0, 0, 0.10], DOWN))
    assert error < 1e-3
    assert len(q) == 7 and np.isfinite(q).all()


def test_the_solution_keeps_the_arm_out_of_the_field(ik, env):
    """The whole point of the collision limit: no solution buries the arm in a bed."""
    weed = env.data.xpos[env.model.body("shepherd_1").id]
    q, _ = ik.solve(env.data.qpos, pose(weed + [0, 0, 0.10], DOWN))
    model = ik.model
    data = mujoco.MjData(model)
    data.qpos[:] = ik._clamped(env.data.qpos)
    data.qpos[ik.arm_qpos] = q
    mujoco.mj_forward(model, data)
    field = [model.geom(n).id for n in ("path", "bed1", "bed2")]
    links = [g for i in range(1, 8)
             for g in range(model.ngeom)
             if model.geom_bodyid[g] == model.body(f"link{i}").id
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    closest = min(mujoco.mj_geomDistance(model, data, a, b, 1.0, None)
                  for a in links for b in field)
    assert closest > CLEARANCE * 0.9


def test_the_base_and_fingers_are_never_moved_by_ik(ik, env):
    """IK owns the seven arm joints only; the rail is commanded separately."""
    before = ik._clamped(env.data.qpos)
    weed = env.data.xpos[env.model.body("shepherd_1").id]
    ik.solve(env.data.qpos, pose(weed + [0, 0, 0.12], DOWN))
    after = ik.configuration.q
    held = [j for j in range(ik.model.njnt) if ik.model.joint(j).name not in ARM]
    rows = ik.model.jnt_qposadr[held]
    assert np.allclose(before[rows], after[rows], atol=1e-9)


def test_an_unreachable_target_reports_its_error_rather_than_raising(ik, env):
    _, error = ik.solve(env.data.qpos, pose([3.0, 0.0, 1.5], DOWN))
    assert error > 0.5


def test_the_fingers_close_horizontally_across_an_upright_stalk(ik, env):
    """A side grasp only grips if the jaws meet across the stem, not along it."""
    for approach in (DOWN, SIDEWAYS, [0.0, -0.87, -0.5]):
        frame = pose([0.2, -0.5, 0.13], approach).rotation().as_matrix()
        assert abs(frame[2, 1]) < 1e-9                   # jaws open in the horizontal plane
        assert frame[2, 0] > -1e-9                       # instrument face is never downward


def test_twisting_turns_the_gripper_about_the_vertical(ik, env):
    """Only rotation about the stalk's own axis reaches the root as torsion."""
    frames = [pose([0.2, -0.5, 0.13], SIDEWAYS, t).rotation().as_matrix()
              for t in (0.0, np.radians(45))]
    for frame in frames:
        assert abs(np.dot(frame[:, 2], [0, 0, 1])) < 1e-9   # approach stays horizontal
    turned = np.degrees(np.arccos(np.clip(np.dot(frames[0][:, 2], frames[1][:, 2]), -1, 1)))
    assert abs(turned - 45) < 1e-6

