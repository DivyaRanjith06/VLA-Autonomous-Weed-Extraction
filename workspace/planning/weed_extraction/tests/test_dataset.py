import collections
import json
import os
import shutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

pytest.importorskip("torch", reason="needs the smolvla extra: run_scripts/setup.sh --smolvla")

from weed_extraction.dataset import build, verify
from weed_extraction.generate import plan_dataset, plan_jobs
from weed_extraction.scene import Manifest

CAMERAS = ["d435i_color", "d435i_depth", "rgb_camera"]


# Planning

def test_the_dataset_plan_covers_every_scene_by_its_split():
    jobs = plan_dataset(0, 4, 1)
    catalogue = Manifest.catalogue(5)
    per_scene = collections.Counter((j["family"], j["variant"]) for j in jobs)
    for family, variants in catalogue.items():
        for m in variants:
            assert per_scene[(family, m.variant)] == (4 if m.split == "train" else 1)
    assert all(j["split"] == catalogue[j["family"]][j["variant"]].split for j in jobs)
    assert not any(j["family"] == Manifest.HELD_OUT_FAMILY for j in jobs if j["split"] == "train")


def test_prompts_are_spread_over_every_kind_the_scenes_offer():
    """Least used first: a kind only a few scenes offer is taken where it is offered."""
    train = [j for j in plan_dataset(0, 4, 1) if j["split"] == "train"]
    kinds = collections.Counter(j["prompt_kind"] for j in train)
    assert set(kinds) >= {"instance", "nearest", "crop_safe", "species_region", "species_area",
                          "species_front", "species_one", "species_all", "side_one"}
    assert kinds["instance"] < 0.5 * len(train)


def test_the_same_seed_plans_the_same_dataset():
    assert plan_dataset(0, 4, 1) == plan_dataset(0, 4, 1)


# Building

def small_plan():
    """Four episodes in two chunks, a training one in each, so the training split is
    merged from two chunk datasets."""
    train = plan_jobs(0, ["midday_dry"], [0], 2)
    jobs = [train[0], *plan_jobs(0, ["evening_cool"], [0], 1),
            train[1], *plan_jobs(0, ["low_light"], [Manifest.HELD_OUT_VARIANT], 1)]
    return {"seed": 0,
            "settings": {"cameras": CAMERAS, "width": 64, "height": 36, "control_hz": 50,
                         "max_steps": 3000},
            "jobs": jobs}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("dataset")
    return out, build(out, small_plan(), chunk_size=2, workers=2, timeout_s=1800)


@pytest.mark.slow
def test_a_chunked_build_merges_one_checked_dataset_per_split(built):
    out, report = built
    assert [c["name"] for c in report["chunks"]] == ["chunk_00", "chunk_01"]
    assert report["splits"]["train"]["chunks"] == 2
    accepted = [json.loads(line) for line in (out / "accepted.jsonl").read_text().splitlines()]
    by_split = collections.Counter(r["split"] for r in accepted)
    assert {s: r["episodes"] for s, r in report["splits"].items()} == dict(by_split)
    for split, result in report["splits"].items():
        assert result["ok"], result["problems"]
        assert result["frames"] == sum(r["frames"] for r in accepted if r["split"] == split)
        assert result["policy_inputs"]["observation.state"] == [28]
        assert result["policy_outputs"] == {"action": [9]}
    assert report["separation"]["ok"]


@pytest.mark.slow
def test_accepted_raw_frames_are_pruned_and_everything_else_kept(built):
    out, _ = built
    for line in (out / "accepted.jsonl").read_text().splitlines():
        record = json.loads(line)
        episode = next((out / "chunks").glob(f"*/episodes/{record['id']}/episode"))
        assert not (episode / "observations.npz").exists()
        for name in ("episode.json", "manifest.json", "validation.json"):
            assert (episode / name).exists()
    for line in (out / "rejected.jsonl").read_text().splitlines():  # kept in full
        record = json.loads(line)
        episode = next((out / "chunks").glob(f"*/episodes/{record['id']}/episode"))
        assert (episode / "observations.npz").exists()


@pytest.mark.slow
def test_a_rerun_resumes_instead_of_regenerating(built):
    out, first = built
    stamps = {p: p.stat().st_mtime_ns for p in (out / "chunks").glob("*/chunk.json")}
    again = build(out, small_plan(), chunk_size=2, workers=2, timeout_s=1800)
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps
    assert again["splits"] == first["splits"]


@pytest.mark.slow
def test_a_different_plan_is_refused(built):
    with pytest.raises(ValueError, match="different plan"):
        build(built[0], small_plan(), chunk_size=3, workers=2, timeout_s=1800)


@pytest.mark.slow
def test_verification_catches_a_merge_that_differs_from_its_chunks(built, tmp_path):
    out, _ = built
    root = tmp_path / "train"
    shutil.copytree(out / "lerobot" / "train", root)
    stats = json.loads((root / "meta/stats.json").read_text())
    stats["action"]["mean"][0] += 0.01
    (root / "meta/stats.json").write_text(json.dumps(stats))
    chunks = sorted((out / "packed").glob("*/train"))
    report = verify(root, chunks)
    assert "stats of action are not the aggregate of its chunks'" in report["problems"]
