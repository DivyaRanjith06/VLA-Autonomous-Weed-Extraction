import json
import os
import shutil

import numpy as np
import pytest

from weed_extraction.generate import check_artifacts, default_workers, generate, plan_jobs
from weed_extraction.prompts import Prompt
from weed_extraction.scene import Manifest


def small_plan(jobs=2):
    """Real episodes, without cameras so they stay quick."""
    return {"seed": 0,
            "settings": {"cameras": [], "width": 64, "height": 36, "control_hz": 50,
                         "max_steps": 3000},
            "jobs": plan_jobs(0, ["midday_dry"], [0], jobs)}


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    """Two episodes generated in parallel by two worker processes."""
    out = tmp_path_factory.mktemp("batch")
    summary = generate(out, small_plan(), workers=2, timeout_s=900)
    return out, summary


def arrays(npz):
    with np.load(npz) as data:
        return {k: data[k] for k in data.files}


# Planning

def test_the_same_seed_plans_the_same_batch():
    assert plan_jobs(3, ["low_light", "midday_loose"], [0, 1], 2) == \
        plan_jobs(3, ["low_light", "midday_loose"], [0, 1], 2)
    assert plan_jobs(3, ["low_light"], [0, 1], 2) != plan_jobs(4, ["low_light"], [0, 1], 2)


def test_every_planned_prompt_is_a_typed_prompt_of_its_scene():
    jobs = plan_jobs(0, list(Manifest.FAMILIES), [0, 1], 2)
    assert len({j["id"] for j in jobs}) == len(jobs)
    catalogue = Manifest.catalogue(2)
    for job in jobs:
        manifest = catalogue[job["family"]][job["variant"]]
        assert (job["scene_seed"], job["split"]) == (manifest.seed, manifest.split)
        assert any(p.text == job["prompt"] and p.targets == (job["target"],)
                   for p in Prompt.typed(manifest))
    for family in Manifest.FAMILIES:  # two episodes per scene, in one wording
        for variant in (0, 1):
            scene = [j for j in jobs if (j["family"], j["variant"]) == (family, variant)]
            assert len(scene) == 2
            species = {p.species for p in catalogue[family][variant].weeds}
            if len(species) == 2:  # the contrast pair: the prompts differ only in the type
                first, second = (j["prompt"] for j in scene)
                assert {j["prompt_kind"] for j in scene} == {"type"}
                assert first.replace("shepherd", "sowthistle") == \
                    second.replace("shepherd", "sowthistle") and first != second


def test_workers_fit_the_free_memory():
    assert 1 <= default_workers(4.0) <= os.cpu_count() - 1
    assert default_workers(1e6) == 1  # never less than one


# Generation

@pytest.mark.slow
def test_a_batch_generates_validates_and_logs_every_episode(batch):
    out, summary = batch
    assert summary["episodes"] == 2 and summary["most_running_at_once"] == 2
    assert summary["accepted"] + summary["rejected"] == 2
    records = [json.loads(line) for name in ("accepted.jsonl", "rejected.jsonl")
               for line in (out / name).read_text().splitlines()]
    assert sorted(r["id"] for r in records) == sorted(j["id"] for j in small_plan()["jobs"])
    for r in records:
        episode = out / "episodes" / r["id"]
        assert (episode / "worker.json").exists()
        assert r["worker"]["returncode"] == 0 and r["worker"]["peak_rss_mb"] > 0
        assert r["failure_labels"] is not None and r["recovery"] in (
            "none", "recovered", "unrecovered")
        saved = json.loads((episode / "episode" / "validation.json").read_text())
        assert saved["accepted"] is r["accepted"] and saved["reason"] == r["reason"]


@pytest.mark.slow
def test_rerunning_the_plan_checks_finished_episodes_instead_of_rerunning_them(batch):
    out, first = batch
    stamps = {p: p.stat().st_mtime_ns for p in (out / "episodes").glob("*/worker.json")}
    again = generate(out, small_plan(), workers=2, timeout_s=900)
    assert {p: p.stat().st_mtime_ns for p in stamps} == stamps
    assert {k: again[k] for k in ("accepted", "rejected", "reasons")} == \
        {k: first[k] for k in ("accepted", "rejected", "reasons")}


@pytest.mark.slow
def test_a_batch_directory_refuses_a_different_plan(batch):
    other = small_plan()
    other["seed"] = 1
    with pytest.raises(ValueError, match="different plan"):
        generate(batch[0], other, workers=1, timeout_s=900)


@pytest.mark.slow
def test_the_same_plan_regenerates_the_same_episodes(batch, tmp_path):
    """Everything but the paths, which name the batch directory."""
    out, _ = batch
    generate(tmp_path, small_plan(), workers=2, timeout_s=900)
    for job in small_plan()["jobs"]:
        a, b = out / "episodes" / job["id"], tmp_path / "episodes" / job["id"]
        assert (a / "episode/manifest.json").read_bytes() == \
            (b / "episode/manifest.json").read_bytes()
        ea, eb = (json.loads((d / "episode/episode.json").read_text()) for d in (a, b))
        assert ea.pop("scene") != eb.pop("scene") and ea == eb
        xa, xb = arrays(a / "episode/observations.npz"), arrays(b / "episode/observations.npz")
        assert xa.keys() == xb.keys()
        assert all(np.array_equal(xa[k], xb[k]) for k in xa)


@pytest.mark.slow
def test_an_unfinished_episode_is_generated_again(batch, tmp_path):
    out = tmp_path / "batch"
    shutil.copytree(batch[0], out)
    victim = out / "episodes" / small_plan()["jobs"][0]["id"]
    (victim / "worker.json").unlink()  # as if the worker died before finishing
    (victim / "episode" / "observations.npz").write_bytes(b"half a file")
    summary = generate(out, small_plan(), workers=2, timeout_s=900)
    assert (victim / "worker.json").exists()
    assert summary["reasons"] == batch[1]["reasons"]


def test_a_worker_that_runs_too_long_is_killed_and_rejected(tmp_path):
    summary = generate(tmp_path, small_plan(jobs=1), workers=1, timeout_s=1)
    assert summary["rejected"] == 1 and summary["reasons"] == {"worker_timeout": 1}
    record = json.loads((tmp_path / "rejected.jsonl").read_text())
    assert record["worker"]["returncode"] != 0


# Artifact checks

@pytest.fixture
def episode(batch, tmp_path):
    """A writable copy of one finished episode, and its job."""
    job = small_plan()["jobs"][0]
    copy = tmp_path / job["id"]
    shutil.copytree(batch[0] / "episodes" / job["id"], copy)
    return copy, job


@pytest.mark.slow
def test_a_finished_episode_passes_the_artifact_checks(episode):
    problems, validation = check_artifacts(*episode, cameras=())
    assert problems == [] and validation is not None


@pytest.mark.slow
@pytest.mark.parametrize("name", ["worker.json", "scene/scene.xml", "episode/episode.json",
                                  "episode/observations.npz", "episode/manifest.json",
                                  "episode/validation.json"])
def test_a_missing_file_is_caught(episode, name):
    (episode[0] / name).unlink()
    problems, _ = check_artifacts(*episode, cameras=())
    assert problems and name in problems[0]


@pytest.mark.slow
def test_a_truncated_archive_is_caught(episode):
    npz = episode[0] / "episode" / "observations.npz"
    npz.write_bytes(npz.read_bytes()[: npz.stat().st_size // 2])
    problems, _ = check_artifacts(*episode, cameras=())
    assert problems and "observations.npz" in problems[0]


@pytest.mark.slow
def test_a_corrupted_archive_member_is_caught(episode):
    npz = episode[0] / "episode" / "observations.npz"
    data = bytearray(npz.read_bytes())
    data[len(data) // 3] ^= 0xFF  # one flipped byte inside a compressed member
    npz.write_bytes(bytes(data))
    problems, _ = check_artifacts(*episode, cameras=())
    assert problems and "observations.npz" in problems[0]


@pytest.mark.slow
def test_a_validation_file_that_disagrees_is_caught(episode):
    path = episode[0] / "episode" / "validation.json"
    saved = json.loads(path.read_text())
    saved["accepted"] = not saved["accepted"]
    path.write_text(json.dumps(saved))
    problems, _ = check_artifacts(*episode, cameras=())
    assert problems == ["the worker's validation.json disagrees with a fresh validation"]


@pytest.mark.slow
def test_an_episode_that_is_not_the_planned_one_is_caught(episode):
    copy, job = episode
    problems, _ = check_artifacts(copy, {**job, "target": "shepherd_99"}, cameras=())
    assert problems and "target" in problems[0]


@pytest.mark.slow
def test_streams_of_different_lengths_are_rejected_by_the_fresh_validation(episode):
    copy, job = episode
    npz = copy / "episode" / "observations.npz"
    data = arrays(npz)
    data["held"] = data["held"][:-1]
    np.savez_compressed(npz, **data)
    problems, validation = check_artifacts(copy, job, cameras=())
    assert problems  # the worker's validation said otherwise
    assert validation.reason == "stream_lengths" and not validation.accepted
