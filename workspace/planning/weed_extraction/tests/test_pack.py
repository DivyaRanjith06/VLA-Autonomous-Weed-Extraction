import json
import os
import shutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # packing and inspection never need the Hub

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the smolvla extra: run_scripts/setup.sh --smolvla")

from weed_extraction.generate import generate, plan_jobs
from weed_extraction.pack import LeRobotDataset, Layout, inspect, pack
from weed_extraction.scene import Manifest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """Two real episodes with all three cameras: the smoke scene that recovers on its
    second attempt, and a held-out-family scene."""
    recovering = {"id": "mixed_clutter_1_0", "family": "mixed_clutter", "variant": 1,
                  "prompt": "remove one shepherd weed from the right bed",
                  "prompt_kind": "species_region", "target": "shepherd_3"}
    held_out = plan_jobs(0, ["evening_cool"], [0], 1)[0]
    scene = Manifest.variants("mixed_clutter", 2)[1]
    recovering.update(scene_seed=scene.seed, split=scene.split)
    plan = {"seed": 0,
            "settings": {"cameras": ["d435i_color", "d435i_depth", "rgb_camera"],
                         "width": 64, "height": 36, "control_hz": 50, "max_steps": 3000},
            "jobs": [recovering, held_out]}
    out = tmp_path_factory.mktemp("batch")
    summary = generate(out, plan, workers=2, timeout_s=1800)
    assert summary["accepted"] == 2, summary
    return out


@pytest.fixture(scope="module")
def packed(batch, tmp_path_factory):
    out = tmp_path_factory.mktemp("packed") / "all"
    return out, pack(batch, out, drop_failed_attempts=False)


def load(root):
    labels = json.loads((root / "meta/weed_extraction/labels.json").read_text())
    return LeRobotDataset(labels["repo_id"], root=root, depth_output_unit="m")


def test_each_split_is_its_own_dataset(packed):
    out, splits = packed
    assert splits == {"train": ["mixed_clutter_1_0"], "held_out_family": ["evening_cool_0_0"]}
    for split in splits:
        info = json.loads((out / split / "meta/info.json").read_text())
        assert info["codebase_version"] == "v3.0" and info["total_episodes"] == 1


def test_the_packed_datasets_pass_inspection(batch, packed):
    out, splits = packed
    for split in splits:
        report = inspect(batch, out / split)
        assert report["ok"], report["problems"]
        assert report["video"]["d435i_depth"]["p99_error_mm"] < 5
        assert all(s["matches_lerobot_recomputation"] for s in report["stats"].values())


def test_frames_follow_the_recording_one_to_one(batch, packed):
    out, _ = packed
    episode = json.loads((batch / "episodes/mixed_clutter_1_0/episode/episode.json").read_text())
    dataset = load(out / "train")
    assert dataset.fps == 50 and dataset.num_frames == episode["frames"]
    assert episode["attempt_count"] == 2  # the failed attempt's frames are all there
    attempts = dataset.select_columns(["label.attempt"]).with_format("numpy")[:]["label.attempt"]
    assert np.bincount(np.ravel(attempts)).tolist() == [
        a["last_frame"] - a["first_frame"] + 1 for a in episode["attempts"]]


def test_dropping_failed_attempts_packs_only_the_final_attempt(batch, tmp_path):
    pack(batch, tmp_path, drop_failed_attempts=True)
    episode = json.loads((batch / "episodes/mixed_clutter_1_0/episode/episode.json").read_text())
    final = episode["attempts"][-1]
    dataset = load(tmp_path / "train")
    assert dataset.num_frames == final["last_frame"] - final["first_frame"] + 1
    attempts = dataset.select_columns(["label.attempt"]).with_format("numpy")[:]["label.attempt"]
    assert set(np.ravel(attempts)) == {final["index"]}
    assert inspect(batch, tmp_path / "train")["ok"]


def test_hidden_metadata_stays_out_of_the_frames(packed):
    out, _ = packed
    info = json.loads((out / "train/meta/info.json").read_text())
    observations = [k for k in info["features"] if k.startswith("observation.")]
    assert sorted(observations) == sorted(k for k in Layout((36, 64), [
        "d435i_color", "d435i_depth", "rgb_camera"]).features if k.startswith("observation."))
    # One state vector, the robot's own: joint positions and velocities, end-effector
    # pose, gripper width -- no weed pose, no root strength.
    assert observations == ["observation.state", *Layout.CAMERAS.values()]
    assert info["features"]["observation.state"]["names"] == list(Layout.STATE)
    assert info["features"]["observation.state"]["shape"] == [28]
    # The root wrench is ground truth no robot sensor gives: a label, never an input.
    assert not any("wrench" in k for k in observations)
    assert info["features"]["label.root_wrench"]["dtype"] == "float32"
    assert info["features"]["label.root_wrench"]["names"] == ["pull_n", "twist_nm"]
    sidecar = [json.loads(line) for line in
               (out / "train/meta/weed_extraction/episodes.jsonl").read_text().splitlines()]
    assert sidecar[0]["source"] == "mixed_clutter_1_0"
    assert sidecar[0]["episode"]["root_break_force_n"] > 0
    assert sidecar[0]["manifest"]["manifest"]["seed"] == sidecar[0]["episode"]["seed"]


def test_inspection_catches_a_frame_count_that_does_not_match(batch, packed, tmp_path):
    root = tmp_path / "train"
    shutil.copytree(packed[0] / "train", root)
    path = root / "meta/weed_extraction/episodes.jsonl"
    record = json.loads(path.read_text())
    record["frames"] += 1
    path.write_text(json.dumps(record) + "\n")
    assert any("frames stored" in p for p in inspect(batch, root)["problems"])


def test_inspection_catches_statistics_that_do_not_match(batch, packed, tmp_path):
    root = tmp_path / "train"
    shutil.copytree(packed[0] / "train", root)
    path = root / "meta/stats.json"
    stats = json.loads(path.read_text())
    stats["action"]["mean"][0] += 0.01
    path.write_text(json.dumps(stats))
    assert "stats of action are not LeRobot's stats of the source" in \
        inspect(batch, root)["problems"]


def test_an_episode_its_validation_rejects_is_never_packed(batch, tmp_path):
    copy = tmp_path / "batch"
    shutil.copytree(batch, copy)
    path = copy / "episodes/evening_cool_0_0/episode/validation.json"
    verdict = json.loads(path.read_text())
    verdict["accepted"] = False
    path.write_text(json.dumps(verdict))
    with pytest.raises(ValueError, match="validation.json rejects it"):
        pack(copy, tmp_path / "out", drop_failed_attempts=False)
