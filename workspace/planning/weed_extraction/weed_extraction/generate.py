"""Parallel episode generation: every episode in its own process, every artifact checked.

    python -m weed_extraction.generate run <batch_dir> [--seed 0 --variants 0 1 ...]

A batch is planned up front into plan.json: which catalogue scenes, which prompt and
target in each, and the recording settings. The plan depends only on its arguments, so
running the same plan again regenerates the same episodes.

The parent never builds a simulator. It launches one short-lived worker process per
episode and runs at most `workers` at once, sized to the free memory. One process per
episode is what keeps memory bounded: a scene's MuJoCo model is never freed in-process
(about 200 MB each), and an episode's frames sit in memory until it is saved. Workers
are plain subprocesses the parent polls, not a pool, so a worker the kernel kills is a
non-zero exit code rather than a hung pool.

A worker writes worker.json last, atomically. An episode without it never finished and
is generated again on the next run; one with it is checked, not rerun. The parent then
checks every artifact -- files present and uncorrupted, a fresh validation agreeing with
the worker's -- and only an episode passing both that and validation is accepted.
"""

import argparse
import dataclasses
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import mujoco
import numpy as np

from weed_extraction.prompts import Prompt
from weed_extraction.recorder import record_episode
from weed_extraction.scene import Manifest, build_env
from weed_extraction.validate import validate_episode


def plan_jobs(seed, families, variants, per_scene, kinds=("type", "type_side")):
    """`per_scene` typed prompts, of `kinds`, in every catalogue scene of `families` x
    `variants`.

    A scene offers up to six: each weed type alone and in each bed. The types alone come
    first, so a scene holding both gives a contrast pair, then the prompts that reach a
    weed no earlier one does; `seed` only breaks ties. Each kind takes the wording used
    least so far across the whole plan.
    """
    catalogue = Manifest.catalogue(max(variants) + 1)
    jobs, used = [], {}
    for family in families:
        family_index = list(Manifest.FAMILIES).index(family)
        for variant in variants:
            manifest = catalogue[family][variant]
            wordings = {}  # (kind, target) -> that prompt in each wording
            for p in (p for p in Prompt.typed(manifest) if p.kind in kinds):
                wordings.setdefault((p.kind, p.targets[0]), []).append(p)
            rng = np.random.default_rng([seed, family_index, variant])
            free = [list(wordings)[i] for i in rng.permutation(len(wordings))]
            chosen = []
            while free and len(chosen) < per_scene:
                taken = {target for _, target in chosen}
                chosen.append(min(free, key=lambda k: (k[0] != "type", k[1] in taken)))
                free.remove(chosen[-1])
            wording = {}  # one per kind in a scene, so a contrast pair differs only in type
            for kind in sorted({kind for kind, _ in chosen}):
                wording[kind] = min(range(len(Prompt.WORDINGS[kind])),
                                    key=lambda i: used.get((kind, i), 0))
                used[kind, wording[kind]] = used.get((kind, wording[kind]), 0) + 1
            for k, key in enumerate(chosen):
                p = wordings[key][wording[key[0]]]
                jobs.append({"id": f"{family}_{variant}_{k}", "family": family,
                             "variant": variant, "scene_seed": manifest.seed,
                             "split": manifest.split, "prompt": p.text, "prompt_kind": p.kind,
                             "target": p.targets[0]})
    return jobs


def plan_dataset(seed, per_scene, layouts, kinds=("type", "type_side")):
    """`per_scene` episodes in each of the first `layouts` training scenes of every
    training family. The held-out family and variant are left for evaluation."""
    known = [f for f in Manifest.FAMILIES if f != Manifest.HELD_OUT_FAMILY]
    variants = [v for v in range(layouts + 1) if v != Manifest.HELD_OUT_VARIANT][:layouts]
    return plan_jobs(seed, known, variants, per_scene, kinds)


def default_workers(per_worker_gb):
    """As many workers as the free memory holds at `per_worker_gb` each, leaving a core.
    Off Linux the free memory is not readable, so half of what is installed counts."""
    if Path("/proc/meminfo").exists():
        with open("/proc/meminfo") as meminfo:
            free_kb = next(int(line.split()[1]) for line in meminfo
                           if line.startswith("MemAvailable"))
    else:
        free_kb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 / 1024
    return max(1, min(os.cpu_count() - 1, int(free_kb / 2**20 // per_worker_gb)))


def generate(batch_dir, plan, workers, timeout_s):
    """Generate every episode of `plan` under `batch_dir`, `workers` at a time, and write
    accepted.jsonl, rejected.jsonl and summary.json. Returns the summary.

    `plan` is {"seed", "settings", "jobs"}. A batch_dir already holding a different plan
    is refused, so two plans never mix; the same plan resumes where it stopped.
    """
    batch_dir = Path(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    plan_file = batch_dir / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError(f"{batch_dir} holds a different plan")
    plan_file.write_text(json.dumps(plan, indent=2))

    start = time.time()
    queue, running, results, peak_running = list(plan["jobs"]), {}, {}, 0
    while queue or running:
        while queue and len(running) < workers:
            job = queue.pop(0)
            out = batch_dir / "episodes" / job["id"]
            if (out / "worker.json").exists():  # finished on an earlier run: check only
                results[job["id"]] = _collect(out, job, plan, returncode=0, timed_out=False)
                continue
            shutil.rmtree(out, ignore_errors=True)  # an unfinished episode starts over
            out.mkdir(parents=True)
            log = open(out / "worker.log", "w")
            proc = subprocess.Popen(
                [sys.executable, "-m", "weed_extraction.generate", "worker",
                 str(batch_dir), job["id"]], stdout=log, stderr=subprocess.STDOUT)
            running[job["id"]] = (proc, job, time.time(), log)
        peak_running = max(peak_running, len(running))
        for job_id, (proc, job, began, log) in list(running.items()):
            timed_out = proc.poll() is None and time.time() - began > timeout_s
            if timed_out:
                proc.kill()
                proc.wait()
            if proc.poll() is not None:
                log.close()
                del running[job_id]
                results[job_id] = _collect(batch_dir / "episodes" / job_id, job, plan,
                                           proc.returncode, timed_out)
        time.sleep(0.2)

    ordered = [results[job["id"]] for job in plan["jobs"]]
    for name, accepted in (("accepted.jsonl", True), ("rejected.jsonl", False)):
        (batch_dir / name).write_text("".join(json.dumps(r) + "\n" for r in ordered
                                              if r["accepted"] is accepted))
    reasons = {}
    for r in ordered:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    peaks = [r["worker"]["peak_rss_mb"] for r in ordered if r["worker"].get("peak_rss_mb")]
    summary = {"episodes": len(ordered), "accepted": sum(r["accepted"] for r in ordered),
               "rejected": sum(not r["accepted"] for r in ordered), "reasons": reasons,
               "workers": workers, "most_running_at_once": peak_running,
               "runtime_s": round(time.time() - start, 1),
               "worker_peak_rss_mb_max": max(peaks, default=None),
               "parent_peak_rss_mb": _peak_rss_mb()}
    (batch_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def check_artifacts(episode_dir, job, cameras):
    """Everything wrong with the files of one finished episode, and its fresh validation.

    The worker's validation.json must agree with a validation run here from the files
    alone; the observations archive is CRC-checked member by member first, so a
    truncated or corrupt file is caught even where a lazy load would not notice.
    """
    episode_dir = Path(episode_dir)
    episode = episode_dir / "episode"
    required = ["worker.json", "scene/scene.xml", "episode/episode.json",
                "episode/observations.npz", "episode/manifest.json", "episode/validation.json"]
    missing = [f for f in required if not (episode_dir / f).exists()]
    if missing:
        return [f"missing {missing}"], None
    problems = []
    try:
        with zipfile.ZipFile(episode / "observations.npz") as archive:
            if (bad := archive.testzip()) is not None:
                problems.append(f"observations.npz member {bad} is corrupt")
    except zipfile.BadZipFile as error:
        problems.append(f"observations.npz is not a readable archive: {error}")
    try:
        saved = json.loads((episode / "validation.json").read_text())
        recorded = json.loads((episode / "episode.json").read_text())
    except ValueError as error:
        return problems + [f"unreadable JSON: {error}"], None
    for key, value in (("target", job["target"]), ("prompt", job["prompt"]),
                       ("family", job["family"]), ("variant", job["variant"]),
                       ("seed", job["scene_seed"])):
        if recorded.get(key) != value:
            problems.append(f"episode {key} {recorded.get(key)!r}, the plan says {value!r}")
    if problems:
        return problems, None
    fresh = validate_episode(episode, cameras=cameras)
    # Compared without the path it was validated at, so a batch can be moved.
    fresh_verdict = json.loads(json.dumps(dataclasses.asdict(fresh)))
    if {**fresh_verdict, "episode": None} != {**saved, "episode": None}:
        problems.append("the worker's validation.json disagrees with a fresh validation")
    return problems, fresh


def run_worker(batch_dir, job_id):
    """Record and validate one planned episode in this process, then mark it finished."""
    batch_dir = Path(batch_dir)
    plan = json.loads((batch_dir / "plan.json").read_text())
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    settings = plan["settings"]
    cameras = tuple(settings["cameras"])
    out = batch_dir / "episodes" / job_id
    began = time.time()
    manifest = Manifest.variants(job["family"], job["variant"] + 1)[job["variant"]]
    prompt = next(p for p in (*Prompt.typed(manifest), *Prompt.valid(manifest))
                  if p.text == job["prompt"] and p.targets == (job["target"],))
    env = build_env(manifest, out / "scene", cameras=cameras, width=settings["width"],
                    height=settings["height"], control_hz=settings["control_hz"],
                    max_steps=settings["max_steps"])
    env.reset()
    if settings.get("parked"):
        park(env, job["target"])
    record_episode(env, manifest, prompt, out / "episode")
    env.close()
    validate_episode(out / "episode", cameras=cameras)
    marker = out / "worker.json.partial"
    marker.write_text(json.dumps({"job": job_id, "runtime_s": round(time.time() - began, 1),
                                  "peak_rss_mb": _peak_rss_mb()}))
    os.replace(marker, out / "worker.json")


def park(env, weed):
    """Drive the base beside `weed` and let it settle, arm at home, before anything is
    recorded: a pick-only episode starts where the scan leaves the robot."""
    model, data = env.model, env.data
    joint = model.joint("base_slide_x")
    target = data.xpos[model.body(weed).id][0]
    data.ctrl[model.actuator("base_x").id] = target
    while (abs(data.qpos[joint.qposadr[0]] - target) > 0.002
           or abs(data.qvel[joint.dofadr[0]]) > 0.005):
        mujoco.mj_step(model, data, nstep=env.n_substeps)


def _collect(out, job, plan, returncode, timed_out):
    """The record of one job once its worker is gone."""
    record = {**job, "accepted": False, "artifact_problems": [], "worker": {}}
    if (out / "worker.json").exists():
        record["worker"] = json.loads((out / "worker.json").read_text())
    record["worker"]["returncode"] = returncode
    if timed_out or returncode:
        tail = (out / "worker.log").read_text().strip().splitlines()[-1:] if (
            out / "worker.log").exists() else []
        record["reason"] = "worker_timeout" if timed_out else "worker_failed"
        record["artifact_problems"] = [f"exit {returncode}: {' '.join(tail)}"]
        return record
    problems, validation = check_artifacts(out, job, tuple(plan["settings"]["cameras"]))
    if problems:
        record.update(reason="artifact_invalid", artifact_problems=problems)
        return record
    episode = json.loads((out / "episode" / "episode.json").read_text())
    record.update(accepted=validation.accepted, reason=validation.reason,
                  final_outcome=episode["final_outcome"], recovery=episode["recovery"],
                  failure_labels=episode["failure_labels"],
                  attempt_count=episode["attempt_count"], frames=episode["frames"])
    return record


def _peak_rss_mb():
    """This process's own peak resident memory. Not getrusage's ru_maxrss: Linux carries
    that across fork and exec, so a worker would report its parent's peak. macOS has
    nothing else, and there it counts bytes."""
    if not Path("/proc/self/status").exists():
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    with open("/proc/self/status") as status:
        return round(next(int(line.split()[1]) for line in status
                          if line.startswith("VmHWM")) / 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="plan and generate a batch")
    run.add_argument("batch_dir", type=Path)
    run.add_argument("--seed", type=int, default=0, help="draws the prompts")
    run.add_argument("--families", nargs="+", default=list(Manifest.FAMILIES))
    run.add_argument("--variants", nargs="+", type=int, default=[0])
    run.add_argument("--per-scene", type=int, default=1, help="episodes per scene")
    run.add_argument("--cameras", nargs="*",
                     default=["d435i_color", "d435i_depth", "rgb_camera", "bed_camera"])
    run.add_argument("--width", type=int, default=384)
    run.add_argument("--height", type=int, default=384)
    # One worker: the environment and IK take about 2 GB, and a worst-case three-attempt
    # episode holds 1428 frames of four cameras besides.
    run.add_argument("--worker-gb", type=float, default=4.0, help="memory per worker")
    run.add_argument("--workers", type=int, help="default: as many as memory allows")
    run.add_argument("--timeout", type=float, default=1800, help="seconds per episode")
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("batch_dir", type=Path)
    worker.add_argument("job_id")
    args = parser.parse_args()

    if args.command == "worker":
        run_worker(args.batch_dir, args.job_id)
        return
    plan = {"seed": args.seed,
            "settings": {"cameras": args.cameras, "width": args.width, "height": args.height,
                         # 25 Hz is near the 30 fps SmolVLA was trained on, and the
                         # expert works at it; 1500 steps cover a three-attempt episode
                         # (at most 1428 frames).
                         "control_hz": 25, "max_steps": 1500},
            "jobs": plan_jobs(args.seed, args.families, args.variants, args.per_scene)}
    workers = args.workers or default_workers(args.worker_gb)
    print(f"{len(plan['jobs'])} episodes, {workers} workers -> {args.batch_dir}", flush=True)
    print(json.dumps(generate(args.batch_dir, plan, workers, args.timeout), indent=2))


if __name__ == "__main__":
    main()
