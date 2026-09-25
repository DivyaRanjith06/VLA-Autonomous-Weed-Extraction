import dataclasses
import json
import shutil

import numpy as np
import pytest

from weed_extraction.prompts import Prompt
from weed_extraction.recorder import record_episode
from weed_extraction.scene import Manifest, build_env
from weed_extraction.validate import log_result, validate_episode

CAMERAS = ("d435i_color", "d435i_depth", "rgb_camera")


@pytest.fixture(scope="module")
def manifest():
    return Manifest.variants("midday_dry", 1)[0]


@pytest.fixture(scope="module")
def prompt(manifest):
    return next(p for p in Prompt.valid(manifest) if len(p.targets) == 1)


@pytest.fixture(scope="module")
def recorded(manifest, prompt, tmp_path_factory):
    """One real episode, recorded without cameras; tests add camera streams as needed."""
    root = tmp_path_factory.mktemp("recorded")
    env = build_env(manifest, root / "scene", cameras=(), control_hz=50, max_steps=1200)
    env.reset()
    record_episode(env, manifest, prompt, root / "episode")
    env.close()
    return root / "episode"


def copy_of(recorded, tmp_path, with_cameras=True):
    """A writable copy of the recorded episode, with fake camera streams of the right
    shape and dtype, so the camera requirement holds unless a test breaks it."""
    episode = tmp_path / "episode"
    shutil.copytree(recorded, episode)
    if with_cameras:
        edit(episode, streams=lambda s, e: s.update(cameras(e)))
    return episode


def cameras(episode, height=4, width=6):
    episode["image_size"] = [height, width]
    frames = episode["frames"]
    return {"d435i_color": np.zeros((frames, height, width, 3), np.uint8),
            "d435i_depth": np.ones((frames, height, width), np.float32),
            "rgb_camera": np.zeros((frames, height, width, 3), np.uint8)}


def edit(episode_dir, streams=None, meta=None):
    """Apply `streams(arrays, episode)` and/or `meta(episode)` to a saved episode."""
    episode = json.loads((episode_dir / "episode.json").read_text())
    arrays = dict(np.load(episode_dir / "observations.npz"))
    if streams:
        streams(arrays, episode)
    if meta:
        meta(episode)
    np.savez_compressed(episode_dir / "observations.npz", **arrays)
    (episode_dir / "episode.json").write_text(json.dumps(episode))


def first(arrays, phase):
    return int(np.flatnonzero(arrays["phase"] == phase)[0])


def reason(episode_dir):
    return validate_episode(episode_dir, cameras=CAMERAS).reason


# The good episode

def test_a_clean_extraction_is_accepted(recorded, tmp_path):
    result = validate_episode(copy_of(recorded, tmp_path), cameras=CAMERAS)
    assert result.accepted and result.reason == "accepted"
    statuses = {c.name: c.status for c in result.checks}
    assert all(s == "pass" for n, s in statuses.items()
               if n not in ("crop_damage", "target_occlusion"))


def test_what_the_simulator_cannot_measure_is_unavailable_not_passed(recorded, tmp_path):
    checks = {c.name: c for c in validate_episode(copy_of(recorded, tmp_path),
                                                  cameras=CAMERAS).checks}
    assert checks["crop_damage"].status == "unavailable"
    assert checks["target_occlusion"].status == "unavailable"
    assert "contact only" in checks["crop_contact"].detail


def test_the_result_is_saved_beside_the_episode(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    result = validate_episode(episode, cameras=CAMERAS)
    saved = json.loads((episode / "validation.json").read_text())
    assert saved["accepted"] is result.accepted and saved["reason"] == result.reason


def test_accepted_and_rejected_episodes_go_to_separate_logs(recorded, tmp_path):
    good = validate_episode(copy_of(recorded, tmp_path / "a"), cameras=CAMERAS)
    bad = validate_episode(copy_of(recorded, tmp_path / "b", with_cameras=False),
                           cameras=CAMERAS)
    log_result(good, tmp_path / "batch")
    log_result(bad, tmp_path / "batch")
    accepted = (tmp_path / "batch" / "accepted.jsonl").read_text().splitlines()
    rejected = (tmp_path / "batch" / "rejected.jsonl").read_text().splitlines()
    assert [json.loads(line)["reason"] for line in accepted] == ["accepted"]
    assert [json.loads(line)["reason"] for line in rejected] == ["streams_present"]


# Streams

def test_a_missing_camera_is_rejected(recorded, tmp_path):
    assert reason(copy_of(recorded, tmp_path, with_cameras=False)) == "streams_present"


def test_a_short_stream_is_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=lambda s, e: s.update(held=s["held"][:-1]))
    assert reason(episode) == "stream_lengths"


@pytest.mark.parametrize("key, bad", [
    ("d435i_color", lambda a: a.astype(np.float32)),       # RGB must be uint8
    ("d435i_depth", lambda a: a[..., None]),                # depth has no channel axis
    ("action", lambda a: a[:, :8]),                         # the 9-D action contract
])
def test_a_misshapen_stream_is_rejected(recorded, tmp_path, key, bad):
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=lambda s, e: s.update({key: bad(s[key])}))
    assert reason(episode) == "stream_shapes"


# Metadata

def test_missing_metadata_is_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, meta=lambda e: e.pop("family"))
    assert reason(episode) == "metadata_complete"


@pytest.mark.parametrize("change", [
    lambda e: e.update(seed=e["seed"] + 1),
    lambda e: e.update(weed_count=e["weed_count"] + 1),
    lambda e: e.update(target="shepherd_99"),
    lambda e: e.update(prompt="remove the dandelion"),
    lambda e: e.update(uprooted_step=None),  # claims failure; the stream says otherwise
])
def test_metadata_that_disagrees_with_the_scene_or_streams_is_rejected(recorded, tmp_path,
                                                                       change):
    episode = copy_of(recorded, tmp_path)
    edit(episode, meta=change)
    assert reason(episode) == "metadata_consistent"


def test_a_root_overridden_at_reset_is_caught(manifest, prompt, tmp_path):
    """Real physics, not an edited file: a caller who resets with a stronger root than
    the scene says has recorded an episode whose hidden metadata would lie."""
    env = build_env(manifest, tmp_path / "scene", cameras=(), control_hz=50, max_steps=1200)
    env.reset(options={"root_strength": {prompt.targets[0]: 1.5}})
    record_episode(env, manifest, prompt, tmp_path / "episode")
    env.close()
    result = validate_episode(tmp_path / "episode", cameras=())
    assert result.reason == "metadata_consistent"
    assert "root thresholds" in next(c.detail for c in result.checks
                                     if c.name == "metadata_consistent")


def test_an_invalid_scene_is_rejected(recorded, tmp_path, manifest):
    episode = copy_of(recorded, tmp_path)
    alike = manifest.strawberries[0].yaw  # two strawberries facing the same way
    turned = dataclasses.replace(manifest, strawberries=(
        manifest.strawberries[0], dataclasses.replace(manifest.strawberries[1], yaw=alike),
        *manifest.strawberries[2:]))
    turned.save(episode / "manifest.json")
    assert reason(episode) == "scene_valid"


# What happened

def test_a_non_finite_value_is_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=lambda s, e: s["pull_n"].__setitem__(5, np.nan))
    assert reason(episode) == "simulation_stable"


def test_a_simulator_warning_is_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, meta=lambda e: e.update(sim_warnings={"mjWARN_BADQACC": 3}))
    assert reason(episode) == "simulation_stable"


def test_phases_out_of_order_are_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=lambda s, e: s["phase"].__setitem__(-1, "reach"))
    assert reason(episode) == "sequence_complete"


def test_a_weed_knocked_loose_on_the_approach_is_rejected(recorded, tmp_path):
    def knock(s, e):
        s["uprooted_this_step"][:] = False
        s["uprooted_this_step"][first(s, "close in")] = True
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=knock)
    assert reason(episode) == "not_uprooted_on_approach"


def test_a_weed_never_grasped_is_rejected(recorded, tmp_path):
    def never(s, e):
        s["held"][:] = False
        s["uprooted_this_step"][:] = False
        e.update(uprooted_step=None, uprooted={})
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=never)
    assert reason(episode) == "target_grasped"


def test_a_grasp_taken_only_during_the_lift_is_rejected(recorded, tmp_path):
    def late(s, e):
        lift = first(s, "lift")
        s["held"][:lift + 1] = False
        s["uprooted_this_step"][:] = False
        s["uprooted_this_step"][lift + 5] = True
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=late)
    assert reason(episode) == "target_grasped"


def test_a_root_that_held_is_rejected(recorded, tmp_path):
    def held(s, e):
        s["uprooted_this_step"][:] = False
        e.update(uprooted_step=None, uprooted={})
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=held)
    assert reason(episode) == "target_uprooted"


def test_a_weed_that_came_loose_while_being_carried_is_rejected(recorded, tmp_path):
    def carried(s, e):
        s["uprooted_this_step"][:] = False
        s["uprooted_this_step"][first(s, "carry")] = True
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=carried)
    assert reason(episode) == "uprooted_by_twist_or_lift"


def test_another_weed_uprooted_is_rejected(recorded, tmp_path, manifest):
    episode = copy_of(recorded, tmp_path)
    target = json.loads((episode / "episode.json").read_text())["target"]
    other = next(p.name for p in manifest.weeds if p.name != target)
    edit(episode, meta=lambda e: e["uprooted"].update({other: 300}))
    assert reason(episode) == "no_other_weed_uprooted"


def test_touching_a_strawberry_is_rejected(recorded, tmp_path):
    def touch(s, e):
        s["crop_contact_n"][400] = 0.8
        e["crop_contacts"] = [{"strawberry": "strawberry_1", "other": "mobile_base",
                               "first_step": 400}]
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=touch)
    result = validate_episode(episode, cameras=CAMERAS)
    assert result.reason == "crop_contact"
    assert "strawberry_1-mobile_base" in next(c.detail for c in result.checks
                                              if c.name == "crop_contact")


def test_letting_go_before_the_drop_is_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, streams=lambda s, e: s["held"].__setitem__(
        slice(first(s, "carry"), None), False))
    assert reason(episode) == "released_in_drop"


def test_a_weed_that_missed_the_bin_is_rejected(recorded, tmp_path):
    episode = copy_of(recorded, tmp_path)
    edit(episode, meta=lambda e: e["final"].update(target_pos=[0.5, -0.5, 0.1]))
    assert reason(episode) == "landed_in_bin"


# The bin floor

def floor(recorded):
    return json.loads((recorded / "episode.json").read_text())["final"]["bin_interior"]


@pytest.mark.parametrize("sink, accepted", [
    (0.0002, True),   # resting on the floor, 0.2 mm into it through soft contact
    (0.004, True),    # still within the 15 mm floor slab
    (0.010, False),   # deeper than any resting weed sinks
])
def test_a_weed_resting_on_the_bin_floor_is_in_the_bin(recorded, tmp_path, sink, accepted):
    box = floor(recorded)
    centre = [(lo + hi) / 2 for lo, hi in zip(box["lo"], box["hi"])]
    episode = copy_of(recorded, tmp_path)
    edit(episode, meta=lambda e: e["final"].update(
        target_pos=[centre[0], centre[1], box["lo"][2] - sink]))
    result = validate_episode(episode, cameras=CAMERAS)
    assert result.accepted is accepted
    if not accepted:
        assert result.reason == "landed_in_bin"


def test_a_weed_on_the_walkway_beside_the_bin_is_not_in_it(recorded, tmp_path):
    box = floor(recorded)
    beside = [box["hi"][0] + 0.05, 0.0, box["lo"][2] - 0.025]  # the walkway, next to it
    episode = copy_of(recorded, tmp_path)
    edit(episode, meta=lambda e: e["final"].update(target_pos=beside))
    assert reason(episode) == "landed_in_bin"
