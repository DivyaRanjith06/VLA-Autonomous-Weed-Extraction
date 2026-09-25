"""Build the training dataset in chunks that fit the disk: generate, pack, check, prune.

    python -m weed_extraction.dataset <out_dir> [--train-per-scene 4 --held-out-per-scene 1]

A raw episode is about 180 MB, so the whole batch does not fit on disk at once. The plan
is split into chunks, and each chunk goes through the usual pipeline -- generated and
validated by `generate`, packed by `pack` and inspected against its raw episodes --
before the raw observations of its accepted episodes are deleted. Everything else is
kept: every episode's metadata, manifest and validation, and rejected episodes in full.
Generation is deterministic, so a pruned episode can be regenerated exactly from its
chunk's plan.

The chunk datasets are then merged, one dataset per split, with LeRobot's own
aggregator, and the merge is checked against them: every non-video value exactly, sampled
video frames exactly (the aggregator copies video streams without re-encoding), frame
counts, statistics, and that no scene reaches two splits.

A finished chunk leaves chunk.json and is skipped on a rerun, so an interrupted build
resumes where it stopped.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from weed_extraction.generate import default_workers, generate, plan_dataset
from weed_extraction.pack import LeRobotDataset, inspect, pack
from weed_extraction.policy import policy_features
from lerobot.datasets.aggregate import aggregate_datasets  # noqa: E402
from lerobot.datasets.compute_stats import aggregate_stats  # noqa: E402


def build(out_dir, plan, chunk_size, workers, timeout_s):
    """Generate, pack, check and prune `plan` chunk by chunk under `out_dir`, then merge
    and check one dataset per split in `out_dir/lerobot`. Returns the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_file = out_dir / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != {**plan, "chunk_size": chunk_size}:
        raise ValueError(f"{out_dir} holds a different plan")
    plan_file.write_text(json.dumps({**plan, "chunk_size": chunk_size}, indent=2))

    jobs = plan["jobs"]
    chunks = []
    for start in range(0, len(jobs), chunk_size):
        name = f"chunk_{start // chunk_size:02d}"
        chunk_plan = {**plan, "jobs": jobs[start:start + chunk_size]}
        chunks.append(_chunk(out_dir, name, chunk_plan, workers, timeout_s))

    for name in ("accepted.jsonl", "rejected.jsonl"):
        (out_dir / name).write_text("".join(
            (out_dir / "chunks" / c["name"] / name).read_text() for c in chunks))
    datasets = {}
    for chunk in chunks:
        for split in chunk["splits"]:
            datasets.setdefault(split, []).append(out_dir / "packed" / chunk["name"] / split)
    report = {"chunks": chunks, "splits": {}}
    for split, roots in datasets.items():
        root = out_dir / "lerobot" / split
        if not (root / "meta" / "weed_extraction" / "verification.json").exists():
            shutil.rmtree(root, ignore_errors=True)  # an unfinished merge starts over
            merge(roots, root, f"weed_extraction/{out_dir.name}_{split}")
        report["splits"][split] = json.loads(
            (root / "meta" / "weed_extraction" / "verification.json").read_text())
    report["separation"] = _separation(out_dir / "lerobot")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def _chunk(out_dir, name, plan, workers, timeout_s):
    """One chunk through generation, packing, inspection and pruning; its record."""
    batch = out_dir / "chunks" / name
    done = batch / "chunk.json"
    if done.exists():
        return json.loads(done.read_text())
    summary = generate(batch, plan, workers, timeout_s)
    packed = out_dir / "packed" / name
    shutil.rmtree(packed, ignore_errors=True)  # an unfinished pack starts over
    accepted = [json.loads(line) for line in (batch / "accepted.jsonl").read_text().splitlines()]
    splits = pack(batch, packed, drop_failed_attempts=False) if accepted else {}
    inspections = {split: inspect(batch, packed / split) for split in splits}
    if not all(r["ok"] for r in inspections.values()):
        raise RuntimeError(f"{name} failed inspection: "
                           f"{ {s: r['problems'] for s, r in inspections.items()} }")
    record = {"name": name, "summary": summary, "splits": splits, "pruned": len(accepted),
              "inspection_ok": True}
    # Marked done before pruning: a rerun must never re-check an episode already pruned,
    # while an interrupted prune only leaves raw frames behind.
    done.write_text(json.dumps(record, indent=2))
    for episode in accepted:  # packed and checked: the raw frames are no longer needed
        (batch / "episodes" / episode["id"] / "episode" / "observations.npz").unlink()
    return record


def merge(roots, root, repo_id):
    """Aggregate the chunk datasets at `roots` into one at `root` and check it against
    them; the check is also written to meta/weed_extraction/verification.json."""
    labels = [json.loads((r / "meta/weed_extraction/labels.json").read_text()) for r in roots]
    aggregate_datasets([l["repo_id"] for l in labels], repo_id, roots=roots, aggr_root=root)
    meta = root / "meta" / "weed_extraction"
    meta.mkdir(parents=True, exist_ok=True)
    sidecar = [json.loads(line) for r in roots
               for line in (r / "meta/weed_extraction/episodes.jsonl").read_text().splitlines()]
    for index, record in enumerate(sidecar):
        record["episode_index"] = index
    (meta / "episodes.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sidecar))
    (meta / "labels.json").write_text(json.dumps(
        {**labels[0], "repo_id": repo_id, "source_batch": [l["source_batch"] for l in labels]},
        indent=2))
    report = verify(root, roots)
    (meta / "verification.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise RuntimeError(f"merged {root} failed verification: {report['problems']}")
    return report


def verify(root, roots):
    """Hold the merged dataset at `root` to the chunk datasets it was made from."""
    labels = json.loads((Path(root) / "meta/weed_extraction/labels.json").read_text())
    merged = LeRobotDataset(labels["repo_id"], root=root, depth_output_unit="m")
    parts = [LeRobotDataset(json.loads((r / "meta/weed_extraction/labels.json").read_text())
                            ["repo_id"], root=r, depth_output_unit="m") for r in roots]
    problems = []
    if merged.num_episodes != sum(p.num_episodes for p in parts):
        problems.append(f"{merged.num_episodes} episodes, parts hold "
                        f"{sum(p.num_episodes for p in parts)}")
    if merged.num_frames != sum(p.num_frames for p in parts):
        problems.append(f"{merged.num_frames} frames, parts hold {sum(p.num_frames for p in parts)}")
    columns = [k for k, f in merged.meta.features.items()
               if f["dtype"] not in ("video", "image", "string")
               and k not in ("index", "episode_index", "task_index")]
    table = merged.select_columns([*columns, "episode_index", "task_index"]).with_format("numpy")[:]
    cameras = [k for k, f in merged.meta.features.items() if f["dtype"] == "video"]
    offset = 0
    for part in parts:
        ptable = part.select_columns([*columns, "episode_index", "task_index"]).with_format("numpy")[:]
        for episode in range(part.num_episodes):
            rows = np.flatnonzero(ptable["episode_index"] == episode)
            mrows = np.flatnonzero(table["episode_index"] == offset + episode)
            if len(rows) != len(mrows):
                problems.append(f"episode {offset + episode}: {len(mrows)} frames, part has {len(rows)}")
                continue
            for key in columns:
                if not np.array_equal(np.stack(table[key][mrows]), np.stack(ptable[key][rows])):
                    problems.append(f"episode {offset + episode}: {key} differs from its chunk")
            if (merged.meta.tasks.index[merged.meta.tasks["task_index"] == table["task_index"][mrows[0]]][0]
                    != part.meta.tasks.index[part.meta.tasks["task_index"] == ptable["task_index"][rows[0]]][0]):
                problems.append(f"episode {offset + episode}: task differs from its chunk")
            for i in (0, len(rows) // 2, len(rows) - 1):
                got, want = merged[int(mrows[i])], part[int(rows[i])]
                for cam in cameras:
                    if not np.array_equal(got[cam].numpy(), want[cam].numpy()):
                        problems.append(f"episode {offset + episode} frame {i}: {cam} differs")
        offset += part.num_episodes
    expected = aggregate_stats([p.meta.stats for p in parts])
    for key, stats in expected.items():
        stored = merged.meta.stats.get(key)
        if stored is None or not all(np.allclose(stored[s], stats[s], rtol=1e-6, atol=1e-7)
                                     for s in stats):
            problems.append(f"stats of {key} are not the aggregate of its chunks'")
    inputs, outputs = policy_features(merged.meta.features)
    return {"episodes": merged.num_episodes, "frames": merged.num_frames, "fps": merged.fps,
            "chunks": len(parts),
            "policy_inputs": {k: list(f.shape) for k, f in inputs.items()},
            "policy_outputs": {k: list(f.shape) for k, f in outputs.items()},
            "features": {k: [f["dtype"], list(f["shape"])] for k, f in merged.meta.features.items()
                         if k.startswith(("observation.", "action", "label."))},
            "problems": problems, "ok": not problems}


def _separation(lerobot_dir):
    """Which scenes each merged split holds; a scene in two splits is a leak."""
    scenes = {}
    for root in sorted(p for p in Path(lerobot_dir).iterdir() if p.is_dir()):
        for line in (root / "meta/weed_extraction/episodes.jsonl").read_text().splitlines():
            episode = json.loads(line)["episode"]
            scenes.setdefault(root.name, set()).add((episode["family"], episode["variant"],
                                                     episode["split"]))
    leaks = {split: sorted({(f, v) for f, v, s in held if s != split})
             for split, held in scenes.items()}
    names = sorted(scenes)
    shared = sorted({scene for i, a in enumerate(names) for b in names[i + 1:]
                     for scene in {(f, v) for f, v, _ in scenes[a]}
                     & {(f, v) for f, v, _ in scenes[b]}})
    return {"scenes_per_split": {s: len(h) for s, h in scenes.items()},
            "episodes_in_wrong_split": {s: l for s, l in leaks.items() if l},
            "scenes_in_several_splits": shared,
            "ok": not any(leaks.values()) and not shared}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="draws the prompts")
    parser.add_argument("--train-per-scene", type=int, default=4)
    parser.add_argument("--held-out-per-scene", type=int, default=1)
    # Fifteen raw episodes are about 2.7 GB.
    parser.add_argument("--chunk-size", type=int, default=15)
    parser.add_argument("--cameras", nargs="*",
                        default=["d435i_color", "d435i_depth", "rgb_camera"])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--worker-gb", type=float, default=4.0, help="memory per worker")
    parser.add_argument("--workers", type=int, help="default: as many as memory allows")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per episode")
    args = parser.parse_args()
    plan = {"seed": args.seed,
            "settings": {"cameras": args.cameras, "width": args.width, "height": args.height,
                         # 50 Hz is the rate the expert is verified at; 3000 steps cover a
                         # three-attempt episode (at most 2850 frames).
                         "control_hz": 50, "max_steps": 3000},
            "jobs": plan_dataset(args.seed, args.train_per_scene, args.held_out_per_scene)}
    workers = args.workers or default_workers(args.worker_gb)
    print(f"{len(plan['jobs'])} episodes in chunks of {args.chunk_size}, {workers} workers "
          f"-> {args.out_dir}", flush=True)
    report = build(args.out_dir, plan, args.chunk_size, workers, args.timeout)
    print(json.dumps({s: {k: r[k] for k in ("episodes", "frames", "ok")}
                      for s, r in report["splits"].items()}, indent=2))
    print(json.dumps(report["separation"], indent=2))


if __name__ == "__main__":
    main()
