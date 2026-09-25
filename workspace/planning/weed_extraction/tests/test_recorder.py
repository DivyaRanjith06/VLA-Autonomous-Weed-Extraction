import dataclasses
import json

import mujoco
import numpy as np
import pytest

from weed_extraction.env import ROOT_BREAK_FORCE, ROOT_BREAK_TORQUE, WeedExtractionEnv
from weed_extraction.ik import ArmIK
from weed_extraction.prompts import Prompt
from weed_extraction.recorder import base_x, ee_pose, gripper_width, in_bin, record_episode
from weed_extraction.scene import Manifest, build_env

SEED = 11  # a scene with single-target prompts for at least one species


def one_target_prompt(manifest, every=False):
    return next(p for p in Prompt.valid(manifest) if len(p.targets) == 1 and p.every == every)


@pytest.fixture(scope="module")
def manifest():
    return Manifest.sample(SEED)


@pytest.fixture
def env(manifest, tmp_path_factory):
    env = build_env(manifest, tmp_path_factory.mktemp("scene"), cameras=(), control_hz=50,
                    max_steps=1200)
    env.reset()
    yield env
    env.close()


def test_a_prompt_naming_more_than_one_weed_is_refused(env, manifest):
    every = next(p for p in Prompt.valid(manifest) if len(p.targets) > 1)
    with pytest.raises(ValueError, match="one episode drives one weed"):
        record_episode(env, manifest, every, "/dev/null")


def test_too_few_max_steps_is_refused_with_a_clear_error(manifest, tmp_path):
    short = build_env(manifest, tmp_path / "short_scene", cameras=(), control_hz=50,
                      max_steps=50)
    short.reset()
    prompt = one_target_prompt(manifest)
    with pytest.raises(RuntimeError, match="raise WeedExtractionEnv"):
        record_episode(short, manifest, prompt, tmp_path / "short")
    short.close()


def test_every_stream_has_one_row_per_frame(env, manifest, tmp_path):
    prompt = one_target_prompt(manifest)
    out = record_episode(env, manifest, prompt, tmp_path)
    episode = json.loads((out / "episode.json").read_text())
    data = np.load(out / "observations.npz")
    frames = episode["frames"]
    assert frames > 0
    for key in ("phase", "action", "uprooted_this_step", "held", "ee_pos", "ee_quat",
                "gripper_width", "base_x", "pull_n", "twist_nm", "qpos", "qvel"):
        assert len(data[key]) == frames, key
    assert data["action"].shape == (frames, 9)
    assert data["qpos"].shape == (frames, env.model.nq)
    assert data["phase"][0] == "reach" and data["phase"][-1] == "drop"


def test_the_episode_carries_the_scene_and_target_metadata(env, manifest, tmp_path):
    prompt = one_target_prompt(manifest)
    out = record_episode(env, manifest, prompt, tmp_path)
    episode = json.loads((out / "episode.json").read_text())
    weed = next(p for p in manifest.weeds if p.name == prompt.targets[0])
    assert episode["prompt"] == prompt.text and episode["target"] == weed.name
    assert episode["seed"] == manifest.seed and episode["family"] == manifest.family
    assert episode["split"] == manifest.split
    assert episode["weed_count"] == len(manifest.weeds)
    assert episode["strawberry_count"] == len(manifest.strawberries)
    assert episode["root_break_force_n"] == pytest.approx(ROOT_BREAK_FORCE * weed.root)
    assert episode["root_break_torque_nm"] == pytest.approx(ROOT_BREAK_TORQUE * weed.root)
    assert Manifest.load(out / "manifest.json") == manifest


def test_the_recorded_observation_caused_the_recorded_action(env, manifest, tmp_path):
    """Replaying qpos[i] with action[i] through the physics must reproduce qpos[i+1],
    proving the observation and action at each index are causally paired -- as long as
    no discrete event (the expert taking or releasing the grasp) falls in that step.
    A mismatched pairing (e.g. action[i+1] instead) would diverge immediately."""
    prompt = one_target_prompt(manifest)
    out = record_episode(env, manifest, prompt, tmp_path)
    data = np.load(out / "observations.npz")
    model, replay = env.model, mujoco.MjData(env.model)
    held = data["held"]
    physics_only = [i for i in range(len(held) - 1) if held[i] == held[i + 1]][:20]
    assert physics_only
    for i in physics_only:
        replay.qpos[:] = data["qpos"][i]
        replay.qvel[:] = data["qvel"][i]
        replay.ctrl[:] = data["action"][i]
        mujoco.mj_step(model, replay, nstep=env.n_substeps)
        assert replay.qpos == pytest.approx(data["qpos"][i + 1], abs=1e-9), i


def test_ee_pose_and_gripper_and_base_readers_match_the_model(env):
    pos, quat = ee_pose(env.model, env.data)
    site = env.model.site("grasp_site").id
    assert pos == pytest.approx(env.data.site_xpos[site])
    expected = np.zeros(4)
    mujoco.mju_mat2Quat(expected, env.data.site_xmat[site])
    assert quat == pytest.approx(expected)
    fingers = [env.model.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")]
    assert gripper_width(env.model, env.data) == pytest.approx(env.data.qpos[fingers].sum())
    rail = env.model.joint("base_slide_x").qposadr[0]
    assert base_x(env.model, env.data) == pytest.approx(env.data.qpos[rail])


def test_cameras_are_recorded_as_configured(manifest, tmp_path):
    env = build_env(manifest, tmp_path / "cam_scene", cameras=("d435i_color", "d435i_depth"),
                    width=64, height=36, control_hz=50, max_steps=1200)
    env.reset()
    prompt = one_target_prompt(manifest)
    out = record_episode(env, manifest, prompt, tmp_path / "cam_episode")
    env.close()
    episode = json.loads((out / "episode.json").read_text())
    data = np.load(out / "observations.npz")
    frames = episode["frames"]
    assert data["d435i_color"].shape == (frames, 36, 64, 3)
    assert data["d435i_color"].dtype == np.uint8
    assert data["d435i_depth"].shape == (frames, 36, 64)
    assert data["d435i_depth"].dtype == np.float32


@pytest.mark.slow
def test_a_successful_extraction_records_a_success_episode(env, manifest, tmp_path):
    prompt = one_target_prompt(manifest)
    out = record_episode(env, manifest, prompt, tmp_path, ik=ArmIK(env.scene))
    episode = json.loads((out / "episode.json").read_text())
    data = np.load(out / "observations.npz")
    assert episode["success"] is True
    assert episode["grasped"] is True and episode["grasped_before_lift"] is True
    assert episode["uprooted_phase"] in ("twist", "lift", "carry")
    assert episode["peak_pull_n"] > 0 and episode["peak_twist_nm"] > 0
    step = episode["uprooted_step"]
    assert data["uprooted_this_step"][step]
    assert not any(data["uprooted_this_step"][:step])


@pytest.mark.parametrize("point, inside", [
    ([0.0, 0.0, 0.05], True),
    ([0.0, 0.0, -0.0002], True),   # sunk 0.2 mm into the floor
    ([0.0, 0.0, -0.006], False),   # past the 5 mm allowance
    ([0.2, 0.0, 0.05], False),     # beyond a wall
    ([0.0, 0.0, 0.2], False),      # above the rim
])
def test_in_bin_allows_a_resting_weed_to_sink_into_the_floor(point, inside):
    assert in_bin(point, [-0.145, -0.114, 0.0], [0.145, 0.114, 0.071]) is inside
