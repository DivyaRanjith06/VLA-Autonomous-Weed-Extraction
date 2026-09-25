import dataclasses
import json
import math
import shutil

import numpy as np
import pytest

from weed_extraction.env import WeedExtractionEnv
from weed_extraction.ik import ArmIK
from weed_extraction.prompts import Prompt
from weed_extraction.recorder import record_episode
from weed_extraction.recovery import Recovery, RecoveryPolicy, RetryUprooter
from weed_extraction.scene import Manifest, build_env
from weed_extraction.uproot import Uprooter
from weed_extraction.validate import validate_episode

# The smoke-batch episode that failed: a fingertip pad settles into the sloped soil and
# the linked fingers stall, so the grasp at 16 mm never closes.
FAILING = ("mixed_clutter", 1, "shepherd_3")


def record(tmp_path, family, variant, weed, policy=RecoveryPolicy(), root=None):
    manifest = Manifest.variants(family, variant + 1)[variant]
    prompt = next(p for p in Prompt.valid(manifest) if p.targets == (weed,))
    env = build_env(manifest, tmp_path / "scene", cameras=(), control_hz=50, max_steps=3000)
    env.reset(options={"root_strength": {weed: root}} if root else None)
    out = record_episode(env, manifest, prompt, tmp_path / "episode", policy=policy)
    env.close()
    return out


def load(episode_dir):
    return (json.loads((episode_dir / "episode.json").read_text()),
            dict(np.load(episode_dir / "observations.npz")))


def edit(episode_dir, streams=None, meta=None):
    episode, arrays = load(episode_dir)
    if streams:
        streams(arrays)
    if meta:
        meta(episode)
    np.savez_compressed(episode_dir / "observations.npz", **arrays)
    (episode_dir / "episode.json").write_text(json.dumps(episode))


def copy_of(episode_dir, tmp_path):
    shutil.copytree(episode_dir, tmp_path / "copy")
    return tmp_path / "copy"


@pytest.fixture(scope="module")
def recovered(tmp_path_factory):
    return record(tmp_path_factory.mktemp("recovered"), *FAILING)


@pytest.fixture(scope="module")
def exhausted(tmp_path_factory):
    """Every retry at the height that failed, so none can succeed."""
    return record(tmp_path_factory.mktemp("exhausted"), *FAILING,
                  policy=RecoveryPolicy(grasp_heights=(0.016,) * 3))


# The tree's decisions, on a real environment but without running the physics

@pytest.fixture(scope="module")
def ik():
    return ArmIK()


@pytest.fixture
def tree(ik):
    env = WeedExtractionEnv(cameras=())
    env.reset()
    yield Recovery(env, "shepherd_1", ik, RecoveryPolicy())
    env.close()


def test_the_first_attempt_is_the_expert_unchanged(tree):
    first = Uprooter(tree.env, "shepherd_1", ik=tree.ik)
    assert tree.runner.GRASP_HEIGHT == first.GRASP_HEIGHT == RecoveryPolicy().grasp_heights[0]
    assert tree.runner.TWIST == first.TWIST
    assert np.array_equal(tree.runner.grip, first.grip)


def test_grasp_failures_climb_the_ladder_until_the_cap(tree):
    assert tree.fail("grasp_failure", "", frame=10)
    assert tree.current.grasp_height_m == 0.020 and tree.runner.GRASP_HEIGHT == 0.020
    assert tree.fail("grasp_failure", "", frame=20)
    assert tree.current.grasp_height_m == 0.024
    assert not tree.fail("grasp_failure", "", frame=30)  # three attempts, the cap
    assert [a.outcome for a in tree.attempts] == ["grasp_failure"] * 3
    assert [(a.first_frame, a.last_frame) for a in tree.attempts] == [(0, 9), (10, 19), (20, 29)]


def test_an_extraction_failure_retries_once_with_more_twist(tree):
    assert tree.fail("extraction_failure", "", frame=10)
    assert tree.current.grasp_height_m == 0.016  # the grasp worked; only the twist changes
    assert tree.current.twist_rad == tree.runner.TWIST == RecoveryPolicy().extraction_twist
    assert not tree.fail("extraction_failure", "", frame=20)


def test_a_weed_knocked_loose_is_not_retried(tree):
    assert not tree.fail("abandoned", "", frame=10)
    assert len(tree.attempts) == 1


@pytest.mark.parametrize("landed, outcome", [(True, "success"), (False, "tray_drop_failure")])
def test_the_end_of_the_sequence_is_success_or_a_tray_drop_failure(tree, landed, outcome):
    tree.finish(frame=950, landed=landed)
    assert [(a.outcome, a.last_frame) for a in tree.attempts] == [(outcome, 949)]


def test_a_retry_lets_go_and_restarts_the_root_clocks(tree):
    env = tree.env
    env.grasp.take(env.data, "shepherd_1")
    env.roots._overloaded_since[tree.root] = 1.0
    tree.fail("extraction_failure", "", frame=10)
    assert env.grasp.held is None and not env.roots._overloaded_since


def test_tray_drop_retries_are_refused_until_a_re_pick_exists():
    with pytest.raises(NotImplementedError, match="re-pick"):
        RecoveryPolicy(tray_drop_retries=1)


def test_retry_uprooter_changes_only_the_grasp_height_and_twist(ik):
    env = WeedExtractionEnv(cameras=())
    env.reset()
    retry = RetryUprooter(env, "shepherd_1", ik, 0.024, math.radians(13))
    plain = Uprooter(env, "shepherd_1", ik=ik)
    assert retry.grip[2] - plain.grip[2] == pytest.approx(0.024 - 0.016)
    assert [n for n, _, _ in retry.phases] == [n for n, _, _ in plain.phases]
    assert Uprooter.GRASP_HEIGHT == 0.016 and Uprooter.TWIST == math.radians(10)  # untouched
    env.close()


# Real physics

@pytest.mark.slow
def test_the_failing_smoke_episode_recovers_on_the_second_attempt(recovered):
    episode, streams = load(recovered)
    attempts = episode["attempts"]
    assert [(a["outcome"], a["grasp_height_m"]) for a in attempts] == [
        ("grasp_failure", 0.016), ("success", 0.020)]
    assert episode["recovery"] == "recovered" and episode["success"]
    assert episode["failure_labels"] == ["grasp_failure"]
    assert np.bincount(streams["attempt"]).tolist() == [
        a["last_frame"] - a["first_frame"] + 1 for a in attempts]
    # The failed attempt stays in the episode, ending as the twist ends.
    assert streams["phase"][attempts[0]["last_frame"]] == "twist"
    assert not streams["uprooted_this_step"][:attempts[1]["first_frame"]].any()
    result = validate_episode(recovered, cameras=())
    assert result.accepted
    assert {c.name: c.status for c in result.checks}["final_outcome"] == "pass"


@pytest.mark.slow
def test_retries_stop_at_the_cap(exhausted):
    episode, streams = load(exhausted)
    assert episode["attempt_count"] == 3 and set(streams["attempt"]) == {0, 1, 2}
    assert episode["failure_labels"] == ["grasp_failure"] * 3
    assert episode["recovery"] == "unrecovered" and not episode["success"]
    assert validate_episode(exhausted, cameras=()).reason == "grasp_failure"


@pytest.mark.slow
def test_a_root_too_strong_to_pull_is_an_extraction_failure(tmp_path):
    out = record(tmp_path, *FAILING, root=50.0)
    episode, streams = load(out)
    labels = [a["outcome"] for a in episode["attempts"]]
    assert "extraction_failure" in labels
    k = labels.index("extraction_failure")
    failed, retry = episode["attempts"][k], episode["attempts"][k + 1]
    assert streams["phase"][failed["last_frame"]] == "lift"
    assert retry["grasp_height_m"] == failed["grasp_height_m"]
    assert retry["twist_rad"] == pytest.approx(RecoveryPolicy().extraction_twist)
    checks = {c.name: c.status for c in validate_episode(out, cameras=()).checks}
    assert checks["attempts_labelled"] == checks["retries_within_policy"] == "pass"


@pytest.mark.slow
def test_a_first_attempt_success_is_recorded_exactly_as_the_expert_ran_it(tmp_path):
    """The recovery tree must not change an episode that needs no recovery: frame for
    frame, the same actions the bare expert takes."""
    manifest = Manifest.variants("midday_dry", 1)[0]
    prompt = next(p for p in Prompt.valid(manifest) if len(p.targets) == 1)
    env = build_env(manifest, tmp_path / "scene", cameras=(), control_hz=50, max_steps=3000)
    env.reset()
    out = record_episode(env, manifest, prompt, tmp_path / "episode")
    env.reset()
    expert, actions = Uprooter(env, prompt.targets[0], ik=ArmIK(env.scene)), []
    while expert.step() is not None:
        actions.append(env.data.ctrl.copy())
        env.step(env.data.ctrl)
    env.close()
    episode, streams = load(out)
    assert np.array_equal(streams["action"], np.array(actions))
    assert episode["recovery"] == "none" and episode["attempt_count"] == 1
    assert not streams["attempt"].any()


# The validator on recovery labels

@pytest.mark.slow
@pytest.mark.parametrize("breakage, why", [
    (dict(meta=lambda e: e["attempts"][0].update(outcome="extraction_failure")),
     "attempts_labelled"),
    (dict(streams=lambda s: s["attempt"].__setitem__(
        np.flatnonzero(s["attempt"] == 1)[0], 0)), "attempts_labelled"),
    (dict(meta=lambda e: e["recovery_policy"].update(max_attempts=1)),
     "retries_within_policy"),
    (dict(meta=lambda e: e["attempts"][1].update(grasp_height_m=0.024)),
     "retries_within_policy"),
    (dict(meta=lambda e: e.update(recovery="none")), "recovery_consistent"),
    (dict(meta=lambda e: e.update(failure_labels=[])), "recovery_consistent"),
])
def test_recovery_labels_that_disagree_with_the_episode_are_rejected(recovered, tmp_path,
                                                                     breakage, why):
    episode = copy_of(recovered, tmp_path)
    edit(episode, **breakage)
    assert validate_episode(episode, cameras=()).reason == why


@pytest.mark.slow
def test_stopping_with_a_retry_left_is_rejected(exhausted, tmp_path):
    episode = copy_of(exhausted, tmp_path)
    edit(episode, meta=lambda e: e["recovery_policy"].update(
        max_attempts=5, grasp_heights=[0.016] * 5))
    assert validate_episode(episode, cameras=()).reason == "retries_within_policy"


@pytest.mark.slow
def test_a_tray_drop_failure_is_the_reason_and_not_retried(recovered, tmp_path):
    def missed(e):
        e["final"]["target_pos"] = [0.5, -0.5, 0.1]
        e["attempts"][-1].update(outcome="tray_drop_failure", detail="not in the bin")
        e.update(final_outcome="tray_drop_failure", success=False, recovery="unrecovered",
                 failure_labels=["grasp_failure", "tray_drop_failure"])
    episode = copy_of(recovered, tmp_path)
    edit(episode, meta=missed)
    result = validate_episode(episode, cameras=())
    assert result.reason == "tray_drop_failure"
    assert {c.name: c.status for c in result.checks}["attempts_labelled"] == "pass"
