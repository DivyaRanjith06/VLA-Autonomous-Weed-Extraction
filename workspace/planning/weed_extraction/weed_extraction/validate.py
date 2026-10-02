"""Accept or reject a recorded episode, with the reason, against the plan's filters.

Every check reads what `record_episode` saved -- observations.npz, episode.json and the
manifest beside them -- and recomputes its own facts from the raw streams rather than
trusting the recorder's summary; where the two disagree, the episode is rejected as
inconsistent. A check the simulator cannot support is reported as unavailable, never
as passed.

Checks run in order and every one that can run is reported -- one that depends on
an earlier failure is left out -- and the first failure is the reason.

An episode may hold several attempts (see weed_extraction.recovery). Each failed
attempt's label is recomputed from the state that ended it, the retries are held to the
policy recorded with them, and the outcome checks -- grasp, uprooting, release, bin --
judge the final attempt alone. Stability, other weeds and crop contact are judged over
the whole episode: what a failed attempt did to the field still counts.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

from weed_extraction.env import ROOT_BREAK_FORCE, ROOT_BREAK_TORQUE
from weed_extraction.prompts import Prompt
from weed_extraction.recorder import STREAMS, in_bin
from weed_extraction.scene import Manifest


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    status: str  # "pass", "fail" or "unavailable"
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class Validation:
    episode: str
    accepted: bool
    reason: str  # "accepted", or the first failed check's name
    checks: tuple[Check, ...]

    def to_json(self):
        return json.dumps(dataclasses.asdict(self))


def validate_episode(episode_dir, cameras=("d435i_color", "d435i_depth", "rgb_camera")):
    """Every check on the episode in `episode_dir`; `cameras` are the ones it must hold.

    The result is also written to `episode_dir/validation.json`.
    """
    episode_dir = Path(episode_dir)
    checks = []
    try:
        episode = json.loads((episode_dir / "episode.json").read_text())
        manifest = Manifest.load(episode_dir / "manifest.json")
        streams = dict(np.load(episode_dir / "observations.npz"))
    except (OSError, ValueError, KeyError) as error:
        checks.append(Check("files", "fail", f"cannot read the episode: {error}"))
    else:
        checks += _streams(episode, streams, cameras)
        if checks[-1].status == "pass":  # the rest index into the streams
            checks += _metadata(episode, manifest, streams)
            checks += _run(episode, streams)
    checks += [
        Check("crop_damage", "unavailable",
              "strawberries are fixed rigid bodies with no damage model; only contact "
              "with them is measured (crop_contact)"),
        Check("target_occlusion", "unavailable",
              "no segmentation stream is recorded, so visibility cannot be measured"),
    ]
    failed = next((c for c in checks if c.status == "fail"), None)
    result = Validation(str(episode_dir), failed is None,
                        "accepted" if failed is None else failed.name, tuple(checks))
    (episode_dir / "validation.json").write_text(json.dumps(dataclasses.asdict(result),
                                                           indent=2))
    return result


def log_result(result, batch_dir):
    """Append `result` to `batch_dir/accepted.jsonl` or `batch_dir/rejected.jsonl`.
    Rejected episodes stay on disk for auditing; only the accepted list feeds training."""
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    name = "accepted.jsonl" if result.accepted else "rejected.jsonl"
    with open(batch_dir / name, "a") as log:
        log.write(result.to_json() + "\n")


def _streams(episode, streams, cameras):
    """Every stream present, one row per frame, the shapes the plan's contract gives."""
    frames = episode.get("frames")
    missing = [k for k in (*STREAMS, "final_qpos", "final_qvel", *cameras) if k not in streams]
    if missing:
        return [Check("streams_present", "fail", f"missing {missing}")]
    checks = [Check("streams_present", "pass")]
    short = {k: len(streams[k]) for k in (*STREAMS, *cameras) if len(streams[k]) != frames}
    if short or not frames:
        return checks + [Check("stream_lengths", "fail", f"{frames} frames, but {short}")]
    checks.append(Check("stream_lengths", "pass", f"{frames} frames"))

    height, width = episode["image_size"]
    wrong = []
    if streams["action"].shape != (frames, 9):  # the plan's 9-D action contract
        wrong.append(f"action {streams['action'].shape}")
    if streams["qpos"].shape[1:] != streams["final_qpos"].shape:
        wrong.append(f"qpos {streams['qpos'].shape} vs final {streams['final_qpos'].shape}")
    if streams["qvel"].shape[1:] != streams["final_qvel"].shape:
        wrong.append(f"qvel {streams['qvel'].shape} vs final {streams['final_qvel'].shape}")
    for cam in cameras:
        image = streams[cam]
        shape, dtype = (((frames, height, width), np.float32) if cam.endswith("_depth")
                        else ((frames, height, width, 3), np.uint8))
        if image.shape != shape or image.dtype != dtype:
            wrong.append(f"{cam} {image.shape} {image.dtype}")
    return checks + [Check("stream_shapes", "fail" if wrong else "pass", "; ".join(wrong))]


def _metadata(episode, manifest, streams):
    """Every field present, and every one agreeing with the manifest and the streams."""
    required = ("prompt", "prompt_kind", "target", "scene", "seed", "family", "variant",
                "split", "weed_count", "strawberry_count", "root_break_force_n",
                "root_break_torque_nm", "frames", "control_hz", "cameras", "image_size",
                "phases", "uprooted", "crop_contacts", "final", "sim_warnings",
                "success", "uprooted_step", "grasped_at_phase", "final_outcome",
                "recovery", "attempt_count", "failure_labels", "attempts",
                "recovery_policy")
    missing = [k for k in required if k not in episode]
    missing += [f"final.{k}" for k in ("target_pos", "bin_interior", "held", "target_rooted")
                if k not in episode.get("final", {})]
    empty = [k for k in required if k in episode and episode[k] is None
             and k not in ("uprooted_step", "grasped_at_phase")]  # None when it never happened
    if missing or empty:
        return [Check("metadata_complete", "fail", f"missing {missing}, empty {empty}")]
    checks = [Check("metadata_complete", "pass")]

    wrong = []
    for key in ("seed", "family", "variant", "split"):
        if episode[key] != getattr(manifest, key):
            wrong.append(f"{key} {episode[key]!r} != manifest {getattr(manifest, key)!r}")
    if (episode["weed_count"], episode["strawberry_count"]) != (
            len(manifest.weeds), len(manifest.strawberries)):
        wrong.append("plant counts differ from the manifest")
    weed = next((p for p in manifest.weeds if p.name == episode["target"]), None)
    if weed is None:
        wrong.append(f"target {episode['target']} is not in the scene")
    elif not (np.isclose(episode["root_break_force_n"], ROOT_BREAK_FORCE * weed.root)
              and np.isclose(episode["root_break_torque_nm"], ROOT_BREAK_TORQUE * weed.root)):
        wrong.append("root thresholds differ from the manifest's root strength")
    if not any(p.text == episode["prompt"] and p.kind == episode["prompt_kind"]
               and p.targets == (episode["target"],)
               for p in (*Prompt.valid(manifest), *Prompt.typed(manifest))):
        wrong.append(f"{episode['prompt']!r} does not name {episode['target']} in this scene")
    if (episode["uprooted_step"] is not None) != bool(streams["uprooted_this_step"].any()):
        wrong.append("the recorded success disagrees with the uprooted stream")
    checks.append(Check("metadata_consistent", "fail" if wrong else "pass", "; ".join(wrong)))

    scene = manifest.problems() + manifest.repeats() + manifest.misses()
    checks.append(Check("scene_valid", "fail" if scene else "pass", "; ".join(scene)))
    return checks


def _run(episode, streams):
    """What happened: stable, sequence complete, grasped, uprooted when and only as it
    should be, crop untouched, released in the drop and landed in the bin."""
    checks = []
    unstable = [k for k, v in streams.items()
                if v.dtype.kind == "f" and not np.isfinite(v).all()]
    if unstable or episode["sim_warnings"]:
        detail = f"non-finite {unstable}; MuJoCo warnings {episode['sim_warnings']}"
        return [Check("simulation_stable", "fail", detail)]
    checks.append(Check("simulation_stable", "pass"))

    recovery = _recovery(episode, streams)
    checks += recovery
    if recovery[0].status == "fail":
        return checks  # without trustworthy attempt boundaries nothing below can be read

    # The outcome checks read the final attempt's frames; frame numbers in their details
    # stay episode-wide.
    first = episode["attempts"][-1]["first_frame"]
    whole, streams = streams, {k: streams[k][first:] for k in STREAMS}
    order = episode["phases"]
    phase = [str(p) for p in streams["phase"]]
    ranks = [order.index(p) if p in order else -1 for p in phase]
    complete = (-1 not in ranks and ranks == sorted(ranks)
                and sorted(set(phase), key=order.index) == order)
    checks.append(Check("sequence_complete", "pass" if complete else "fail",
                        "" if complete else f"phases ran {sorted(set(phase))}"))
    if not complete:
        return checks

    # `held` is observed before each action, so the grasp taken during frame i's action
    # first shows at i+1; likewise the release.
    held = streams["held"].astype(bool)
    grasp = next((i for i in range(1, len(held)) if held[i] and not held[i - 1]), None)
    loose = np.flatnonzero(streams["uprooted_this_step"])
    step = int(loose[0]) if len(loose) else None

    early = step is not None and (grasp is None or step < grasp)
    checks.append(Check("not_uprooted_on_approach", "fail" if early else "pass",
                        f"came loose in {phase[step]} before any grasp" if early else ""))

    grasped_in = phase[grasp - 1] if grasp is not None else None
    if grasped_in is None:
        checks.append(Check("target_grasped", "fail", "the fingers never closed on the stem"))
    elif order.index(grasped_in) >= order.index("lift"):
        checks.append(Check("target_grasped", "fail", f"grasped only in {grasped_in}"))
    else:
        checks.append(Check("target_grasped", "pass", f"grasped in {grasped_in}"))

    if step is None:
        checks.append(Check("target_uprooted", "fail",
                            f"the root held; peak pull {streams['pull_n'].max():.1f} N, "
                            f"twist {streams['twist_nm'].max():.3f} N.m"))
    else:
        checks.append(Check("target_uprooted", "pass", f"at step {first + step}"))
        if not early:
            proper = phase[step] in ("twist", "lift")
            checks.append(Check("uprooted_by_twist_or_lift", "pass" if proper else "fail",
                                f"in {phase[step]}"))

    others = sorted(w for w in episode["uprooted"] if w != episode["target"])
    checks.append(Check("no_other_weed_uprooted", "fail" if others else "pass",
                        f"also uprooted {others}" if others else ""))

    touched = episode["crop_contacts"]
    checks.append(Check(
        "crop_contact", "fail" if touched else "pass",
        (f"{len(touched)} strawberry contact(s), peak {whole['crop_contact_n'].max():.2f} N: "
         + ", ".join(f"{c['strawberry']}-{c['other']}" for c in touched[:4]))
        if touched else "no strawberry touched (contact only: damage is not modelled)"))

    if grasp is not None:
        release = next((i for i in range(grasp + 1, len(held)) if not held[i]), None)
        released_in = phase[release - 1] if release is not None else None
        checks.append(Check("released_in_drop", "pass" if released_in == "drop" else "fail",
                            f"let go in {released_in}" if released_in else "never let go"))

    if step is not None:
        box = episode["final"]["bin_interior"]
        end = episode["final"]["target_pos"]
        inside = in_bin(end, box["lo"], box["hi"])
        checks.append(Check("landed_in_bin", "pass" if inside else "fail",
                            f"stem base ended at {np.round(end, 3).tolist()}"))
    return checks


def _recovery(episode, streams):
    """The attempts: labelled as the streams show, retried as the policy allows, and
    summed up consistently. The first check gates the rest."""
    attempts, policy = episode["attempts"], episode["recovery_policy"]
    frames, labels = episode["frames"], streams["attempt"]
    phase, held = [str(p) for p in streams["phase"]], streams["held"].astype(bool)
    uprooted = streams["uprooted_this_step"].astype(bool)
    final = episode["final"]

    wrong = []
    bounds = [(a["first_frame"], a["last_frame"]) for a in attempts]
    if ([a["index"] for a in attempts] != list(range(len(attempts)))
            or bounds[0][0] != 0 or bounds[-1][1] != frames - 1
            or any(b[0] != a[1] + 1 for a, b in zip(bounds, bounds[1:]))):
        wrong.append(f"attempt frame ranges {bounds} do not tile {frames} frames")
    elif not np.array_equal(labels, np.repeat(np.arange(len(attempts)),
                                              [b - a + 1 for a, b in bounds])):
        wrong.append("the attempt stream disagrees with the attempt frame ranges")
    else:
        for k, attempt in enumerate(attempts):
            if attempt["outcome"] == "success":
                continue  # a success is judged in full by the outcome checks
            start, end = bounds[k]
            # What the attempt ended on: the next attempt's first snapshot, or for the
            # last attempt the terminal state.
            last = k == len(attempts) - 1
            held_after = bool(final["held"]) if last else bool(held[end + 1])
            rooted_after = not uprooted[:end + 1].any()
            box = final["bin_interior"]
            landed = in_bin(final["target_pos"], box["lo"], box["hi"])
            if phase[end] == "twist" and not held_after:
                expected = "grasp_failure" if rooted_after else "abandoned"
            elif phase[end] == "lift" and held_after and rooted_after:
                expected = "extraction_failure"
            elif phase[end] == "drop" and last and not landed:
                expected = "tray_drop_failure"
            else:
                expected = f"nothing (ended in {phase[end]}, held {held_after}, "
                expected += f"rooted {rooted_after})"
            if attempt["outcome"] != expected:
                wrong.append(f"attempt {k} labelled {attempt['outcome']}, streams show "
                             f"{expected}")
    checks = [Check("attempts_labelled", "fail" if wrong else "pass", "; ".join(wrong))]
    if wrong:
        return checks

    wrong = []
    retried = [a["outcome"] for a in attempts[:-1]]
    if len(attempts) > policy["max_attempts"]:
        wrong.append(f"{len(attempts)} attempts, cap {policy['max_attempts']}")
    if set(retried) - {"grasp_failure", "extraction_failure"}:
        wrong.append(f"retried after {sorted(set(retried))}")
    if retried.count("grasp_failure") > len(policy["grasp_heights"]) - 1:
        wrong.append("more grasp retries than grasp heights")
    if retried.count("extraction_failure") > policy["extraction_retries"]:
        wrong.append("more extraction retries than allowed")
    for k, attempt in enumerate(attempts):
        before = [a["outcome"] for a in attempts[:k]]
        height = policy["grasp_heights"][min(before.count("grasp_failure"),
                                             len(policy["grasp_heights"]) - 1)]
        twist = (policy["extraction_twist"] if "extraction_failure" in before
                 else attempts[0]["twist_rad"])
        if not (np.isclose(attempt["grasp_height_m"], height)
                and np.isclose(attempt["twist_rad"], twist)):
            wrong.append(f"attempt {k} ran at {attempt['grasp_height_m']} m, "
                         f"{attempt['twist_rad']} rad; the policy gives {height} m, {twist} rad")
    end = attempts[-1]["outcome"]
    room = len(attempts) < policy["max_attempts"]
    if room and ((end == "grasp_failure"
                  and retried.count("grasp_failure") + 1 < len(policy["grasp_heights"]))
                 or (end == "extraction_failure"
                     and retried.count("extraction_failure") < policy["extraction_retries"])):
        wrong.append(f"stopped after {end} with a retry left")
    checks.append(Check("retries_within_policy", "fail" if wrong else "pass", "; ".join(wrong)))

    success = end == "success"
    expected = {"final_outcome": end, "success": success,
                "failure_labels": [a["outcome"] for a in attempts if a["outcome"] != "success"],
                "attempt_count": len(attempts),
                "recovery": ("unrecovered" if not success
                             else "recovered" if len(attempts) > 1 else "none")}
    wrong = [f"{k} {episode[k]!r}, attempts give {v!r}" for k, v in expected.items()
             if episode[k] != v]
    checks.append(Check("recovery_consistent", "fail" if wrong else "pass", "; ".join(wrong)))

    # Named after the failure itself, so an unrecovered episode's reason says what failed.
    if success:
        checks.append(Check("final_outcome", "pass",
                            f"succeeded on attempt {len(attempts)}"))
    else:
        checks.append(Check(end, "fail", f"{attempts[-1]['detail']}, after "
                                         f"{len(attempts)} attempt(s)"))
    return checks
