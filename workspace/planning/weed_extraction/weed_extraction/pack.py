"""Pack a generated batch's accepted episodes into LeRobot Dataset v3, and inspect it.

    python -m weed_extraction.pack <batch_dir> <out_dir> [--drop-failed-attempts]

One dataset per split -- train, held_out_family, held_out_variant -- so held-out scenes
never reach training by accident. Each episode is packed as recorded, at the control
rate it was recorded at, frame for frame: the observation that caused an action stays
beside that action.

What the policy may see is under `observation.*`: only what the robot itself could
sense, its proprioception folded into the one `observation.state` vector SmolVLA reads. The per-frame expert labels (phase, attempt, held, uprooted) and the root wrench
are under `label.*`, for evaluation and analysis -- the wrench is the simulator's
ground-truth pull and twist at the root, which no sensor on the robot measures. The
scene's hidden metadata -- root strengths, the full manifest, recovery history -- is
kept beside the dataset in meta/weed_extraction/, never in a frame.

Inspection reads the dataset back through LeRobot and compares it with the source
artifacts: shapes, frame counts, every non-video value exactly, sampled video frames
within the codec's error, and the stored statistics against LeRobot's own computation
run on the source frames.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from weed_extraction.env import SCENE

try:
    import lerobot  # noqa: F401
except ModuleNotFoundError:  # not installed: use the vendored submodule's source
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lerobot" / "src"))
from lerobot.configs.video import depth_encoder_defaults  # noqa: E402
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


class Layout:
    """How the recorded streams map onto dataset features."""

    CAMERAS = {"d435i_color": "observation.images.wrist",
               "d435i_depth": "observation.images.wrist_depth",
               "rgb_camera": "observation.images.overview"}
    # The robot's joints, in the order the state lists them.
    JOINTS = ("base_slide_x", *(f"joint{i}" for i in range(1, 8)), "finger_joint1",
              "finger_joint2")
    ACTIONS = (*(f"joint{i}" for i in range(1, 8)), "gripper", "base_x")
    # Joint positions, joint velocities, the end-effector pose and the gripper width.
    STATE = (*(f"q.{j}" for j in JOINTS), *(f"qd.{j}" for j in JOINTS),
             "ee.x", "ee.y", "ee.z", "ee.qw", "ee.qx", "ee.qy", "ee.qz", "gripper_width")
    LABELS = ("label.phase", "label.attempt", "label.held", "label.uprooted")  # integers

    def __init__(self, image_size, cameras):
        # The robot is included first in every scene, so its joints sit alike in each.
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.qpos = [int(model.joint(j).qposadr[0]) for j in self.JOINTS]
        self.qvel = [int(model.joint(j).dofadr[0]) for j in self.JOINTS]
        self.cameras = list(cameras)
        height, width = image_size
        self.features = {
            "observation.state": _vector(self.STATE),
            "action": _vector(self.ACTIONS),
            "label.root_wrench": _vector(("pull_n", "twist_nm")),
            **{label: {"dtype": "int64", "shape": (1,), "names": None} for label in self.LABELS},
        }
        for cam in self.cameras:
            depth = cam.endswith("_depth")
            self.features[self.CAMERAS[cam]] = {
                "dtype": "video", "shape": (height, width, 1 if depth else 3),
                "names": ["height", "width", "channels"], "info": {"is_depth_map": depth}}

    def frames(self, streams, phases, task):
        """Every recorded frame as a LeRobot frame dict."""
        index = {p: i for i, p in enumerate(phases)}
        for i in range(len(streams["action"])):
            frame = {
                "observation.state": np.concatenate([
                    streams["qpos"][i][self.qpos], streams["qvel"][i][self.qvel],
                    streams["ee_pos"][i], streams["ee_quat"][i],
                    streams["gripper_width"][i:i + 1]]),
                "action": streams["action"][i],
                "label.root_wrench": np.array([streams["pull_n"][i], streams["twist_nm"][i]]),
                "label.phase": [index[str(streams["phase"][i])]],
                "label.attempt": [streams["attempt"][i]],
                "label.held": [streams["held"][i]],
                "label.uprooted": [streams["uprooted_this_step"][i]],
                "task": task,
            }
            frame = {k: v if k == "task" else np.asarray(
                v, np.int64 if k in self.LABELS else np.float32) for k, v in frame.items()}
            for cam in self.cameras:
                image = streams[cam][i]
                frame[self.CAMERAS[cam]] = image[..., None] if cam.endswith("_depth") else image
            yield frame


def pack(batch_dir, out_dir, drop_failed_attempts):
    """Pack every accepted episode of `batch_dir` into `out_dir/<split>`. Returns the
    splits written and, for each, the source records packed in episode order."""
    batch_dir, out_dir = Path(batch_dir), Path(out_dir)
    plan = json.loads((batch_dir / "plan.json").read_text())
    records = [json.loads(line) for line in
               (batch_dir / "accepted.jsonl").read_text().splitlines()]
    splits = {}
    for record in records:
        splits.setdefault(record["split"], []).append(record)
    settings = plan["settings"]
    layout = Layout((settings["height"], settings["width"]), settings["cameras"])
    for split, members in splits.items():
        root = out_dir / split
        repo_id = f"weed_extraction/{batch_dir.name}_{split}"
        dataset = LeRobotDataset.create(
            repo_id, fps=settings["control_hz"],
            features=layout.features, root=root, robot_type="mobile_panda",
            image_writer_threads=4)
        sidecar = []
        for record in members:
            episode_dir = batch_dir / "episodes" / record["id"] / "episode"
            episode = json.loads((episode_dir / "episode.json").read_text())
            if not json.loads((episode_dir / "validation.json").read_text())["accepted"]:
                raise ValueError(f"{record['id']} is in accepted.jsonl but its "
                                 "validation.json rejects it")
            with np.load(episode_dir / "observations.npz") as archive:
                streams = {k: archive[k] for k in archive.files}  # decompressed once each
            first = episode["attempts"][-1]["first_frame"] if drop_failed_attempts else 0
            streams = {k: v[first:] for k, v in streams.items() if not k.startswith("final_")}
            for frame in layout.frames(streams, episode["phases"], episode["prompt"]):
                dataset.add_frame(frame)
            dataset.save_episode()
            sidecar.append({"episode_index": len(sidecar), "source": record["id"],
                            "frames": len(streams["action"]), "first_source_frame": first,
                            "episode": episode,
                            "manifest": json.loads((episode_dir / "manifest.json").read_text())})
        dataset.finalize()
        meta = root / "meta" / "weed_extraction"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "episodes.jsonl").write_text("".join(json.dumps(s) + "\n" for s in sidecar))
        (meta / "labels.json").write_text(json.dumps({
            "repo_id": repo_id, "label.phase": episode["phases"],
            "drop_failed_attempts": drop_failed_attempts, "source_batch": str(batch_dir),
            "cameras": {c: layout.CAMERAS[c] for c in layout.cameras}}, indent=2))
    return {split: [r["id"] for r in members] for split, members in splits.items()}


def inspect(batch_dir, root):
    """Read the packed dataset at `root` back through LeRobot and hold it to its source
    episodes. Returns the report, also written to meta/weed_extraction/inspection.json."""
    batch_dir, root = Path(batch_dir), Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    sidecar = [json.loads(line) for line in
               (root / "meta" / "weed_extraction" / "episodes.jsonl").read_text().splitlines()]
    labels = json.loads((root / "meta" / "weed_extraction" / "labels.json").read_text())
    dataset = LeRobotDataset(labels["repo_id"], root=root, depth_output_unit="m")
    plan = json.loads((batch_dir / "plan.json").read_text())
    layout = Layout((plan["settings"]["height"], plan["settings"]["width"]),
                    plan["settings"]["cameras"])
    problems = []

    if info["codebase_version"] != "v3.0":
        problems.append(f"codebase_version {info['codebase_version']}")
    for key, feature in layout.features.items():
        stored = info["features"].get(key)
        if stored is None or tuple(stored["shape"]) != tuple(feature["shape"]) \
                or stored["dtype"] != feature["dtype"]:
            problems.append(f"feature {key}: stored {stored and (stored['dtype'], stored['shape'])}"
                            f", expected {(feature['dtype'], feature['shape'])}")
    if dataset.num_episodes != len(sidecar):
        problems.append(f"{dataset.num_episodes} episodes, {len(sidecar)} packed")

    columns = [k for k, f in layout.features.items() if f["dtype"] != "video"]
    table = dataset.select_columns([*columns, "episode_index", "frame_index", "timestamp",
                                    "task_index"]).with_format("numpy")[:]
    source_all = {k: [] for k in columns}
    episode_stats = []
    enc = depth_encoder_defaults()
    video_errors = {cam: [] for cam in layout.cameras}
    for s in sidecar:
        k = s["episode_index"]
        rows = np.flatnonzero(table["episode_index"] == k)
        episode_dir = batch_dir / "episodes" / s["source"] / "episode"
        with np.load(episode_dir / "observations.npz") as archive:
            streams = {key: archive[key] for key in archive.files}
        streams = {key: v[s["first_source_frame"]:] for key, v in streams.items()
                   if not key.startswith("final_")}
        expected = list(layout.frames(streams, s["episode"]["phases"], s["episode"]["prompt"]))
        if len(rows) != len(expected) or len(rows) != s["frames"]:
            problems.append(f"episode {k}: {len(rows)} frames stored, {len(expected)} recorded")
            continue
        if not np.array_equal(table["frame_index"][rows], np.arange(len(rows))) or \
                not np.allclose(table["timestamp"][rows], np.arange(len(rows)) / dataset.fps):
            problems.append(f"episode {k}: frame indices or timestamps are not contiguous")
        tasks = dataset.meta.tasks
        if tasks.index[tasks["task_index"] == table["task_index"][rows[0]]][0] != \
                s["episode"]["prompt"]:
            problems.append(f"episode {k}: task is not its prompt")
        for key in columns:
            want = np.stack([f[key] for f in expected])
            got = np.stack(table[key][rows]).reshape(want.shape)
            source_all[key].append(want)
            if not np.array_equal(got, want):
                problems.append(f"episode {k}: {key} differs from the source")
        episode_stats.append(compute_episode_stats(
            {key: source_all[key][-1] for key in columns},
            {key: layout.features[key] for key in columns}))
        start = int(rows[0])
        for i in (0, len(rows) // 2, len(rows) - 1):  # first, middle and last frame
            item = dataset[start + i]
            for cam in layout.cameras:
                got = item[layout.CAMERAS[cam]].numpy()
                if cam.endswith("_depth"):
                    src = expected[i][layout.CAMERAS[cam]][..., 0]
                    inside = src <= enc.depth_max
                    video_errors[cam].append(np.abs(got[0][inside] - src[inside]))
                    if not np.allclose(got[0][~inside], enc.depth_max, atol=1e-3):
                        problems.append(f"episode {k} frame {i}: depth beyond "
                                        f"{enc.depth_max} m is not clipped to it")
                else:
                    src = expected[i][layout.CAMERAS[cam]].astype(np.float32)
                    video_errors[cam].append(np.abs(got.transpose(1, 2, 0) * 255 - src).ravel())

    report = {"root": str(root), "codebase_version": info["codebase_version"],
              "fps": dataset.fps, "episodes": dataset.num_episodes,
              "frames": dataset.num_frames, "features": {
                  k: [info["features"][k]["dtype"], info["features"][k]["shape"]]
                  for k in layout.features},
              "video": {}, "stats": {}}
    for cam, errors in video_errors.items():
        if not errors:
            continue
        e = np.concatenate(errors)
        if cam.endswith("_depth"):
            report["video"][cam] = {"median_error_mm": round(float(np.median(e)) * 1000, 3),
                                    "p99_error_mm": round(float(np.percentile(e, 99)) * 1000, 3)}
            if np.percentile(e, 99) > 0.005:
                problems.append(f"{cam}: depth p99 error {np.percentile(e, 99) * 1000:.1f} mm")
        else:
            report["video"][cam] = {"mean_abs_error_of_255": round(float(e.mean()), 3),
                                    "psnr_db": round(float(20 * np.log10(255 / np.sqrt(
                                        np.mean(e ** 2)))), 2)}
            if e.mean() > 8:
                problems.append(f"{cam}: mean error {e.mean():.1f} of 255")
    # The stored stats must be what LeRobot computes from the source frames. Against
    # float64 they drift a little, which is reported, not failed: LeRobot takes the
    # variance as E[x^2] - E[x]^2 in float32, which cancels badly on a channel whose mean
    # is large beside its spread.
    if len(episode_stats) != len(sidecar):  # the frames must match before their stats can
        problems.append("statistics not checked: not every episode's frames match its source")
        columns = []
    else:
        expected_stats = aggregate_stats(episode_stats)
    for key in columns:
        values = np.concatenate(source_all[key]).astype(np.float64)
        stored = dataset.meta.stats[key]
        count = int(np.asarray(stored["count"]).ravel()[0])
        matches = count == len(values) and all(
            np.allclose(stored[stat], expected_stats[key][stat], rtol=1e-6, atol=1e-7)
            for stat in expected_stats[key])
        if not matches:
            problems.append(f"stats of {key} are not LeRobot's stats of the source")
        drift = max(float(np.max(np.abs(np.asarray(stored[stat]) - exact)
                                 / np.maximum(1e-6, np.abs(exact))))
                    for stat in ("mean", "std", "min", "max")
                    for exact in [getattr(np, stat)(values, 0)])
        report["stats"][key] = {"count": count, "matches_lerobot_recomputation": matches,
                                "worst_relative_drift_from_float64": drift}
    for cam in layout.cameras:
        stats = dataset.meta.stats.get(layout.CAMERAS[cam])
        if stats is None or np.asarray(stats["mean"]).shape != ((1, 1, 1) if cam.endswith(
                "_depth") else (3, 1, 1)):
            problems.append(f"no per-channel stats for {layout.CAMERAS[cam]}")
    report["problems"] = problems
    report["ok"] = not problems
    (root / "meta" / "weed_extraction" / "inspection.json").write_text(
        json.dumps(report, indent=2))
    return report


def _vector(names):
    return {"dtype": "float32", "shape": (len(names),), "names": list(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--drop-failed-attempts", action="store_true",
                        help="pack only each episode's final attempt")
    args = parser.parse_args()
    if args.out_dir.exists():
        sys.exit(f"{args.out_dir} exists; packing writes fresh datasets only")
    for split in pack(args.batch_dir, args.out_dir, args.drop_failed_attempts):
        print(json.dumps(inspect(args.batch_dir, args.out_dir / split), indent=2))


if __name__ == "__main__":
    main()
