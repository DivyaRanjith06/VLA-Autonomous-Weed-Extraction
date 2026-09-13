"""Launch file: `weed-launch <config> [--key=value ...]`.

<config> is a YAML path or the name of one in configs/ (e.g. `weed-launch view`).
Any config field can be overridden on the command line, e.g. `--base_x=1.5`.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import mujoco
import numpy as np

from weed_extraction.env import (ROOT_BREAK_FORCE, ROOT_BREAK_TORQUE, SCENE,
                                 WeedExtractionEnv)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

DRAG_TRAVEL = 0.025  # metres of drag that produces ROOT_BREAK_FORCE
DRAG_STIFFNESS = ROOT_BREAK_FORCE / DRAG_TRAVEL
DRAG_FORCE_CAP = 1.3 * ROOT_BREAK_FORCE  # a flick cannot hugely overshoot the root


def drag_force(model, data, pert):
    """The pull a mouse drag puts on a weed: a critically damped spring to the cursor."""
    body = pert.select
    stretch = pert.refselpos - data.xipos[body]
    damping = 2 * np.sqrt(DRAG_STIFFNESS * model.body_mass[body])  # critical: no ringing
    force = DRAG_STIFFNESS * stretch - damping * data.cvel[body, 3:]
    size = np.linalg.norm(force)
    return force * (DRAG_FORCE_CAP / size) if size > DRAG_FORCE_CAP else force


@dataclass
class LaunchConfig:
    mode: str = "view"  # view: interactive viewer; capture: one frame per camera; uproot: scripted
    scene: Path = SCENE
    cameras: list[str] = field(default_factory=lambda: ["d435i_color", "d435i_depth"])  # capture only
    width: int = 640
    height: int = 360
    base_x: float = 0.0  # base target along the bed, metres
    control_hz: int = 20  # how often the controller writes actuator targets
    weed: str = "shepherd_1"  # uproot: which weed to pull
    settle_s: float = 2.0  # capture: simulated seconds before the frame is taken
    supersample: int = 2  # capture: render this many times larger, then shrink
    output_dir: Path = Path("outputs")


def make_env(cfg, cameras, scale=1):
    env = WeedExtractionEnv(cfg.scene, cameras, cfg.width * scale, cfg.height * scale,
                            control_hz=cfg.control_hz)
    env.reset()
    env.data.ctrl[env.model.actuator("base_x").id] = cfg.base_x
    return env


def to_rgb(image):
    """Depth (metres) becomes a colour map; RGB passes through."""
    if image.ndim == 3:
        return image
    scaled = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def view(cfg):
    """Interactive viewer. Ctrl+right-drag a weed to pull it out of the soil."""
    import mujoco.viewer

    # No camera rendering here: an offscreen frame of this scene costs ~200 ms,
    # which is what kept the viewer from running in real time.
    env = make_env(cfg, cameras=[])
    m, d, roots = env.model, env.data, env.roots
    # One frame is many physics steps: syncing the viewer every 1 ms step is far
    # slower than the physics itself, and the window would crawl behind real time.
    substeps = max(1, round(1 / 60 / m.opt.timestep))
    next_readout = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        pert = viewer.perturb
        while viewer.is_running():
            frame_start = time.time()
            pulling, pull = None, 0.0
            for _ in range(substeps):
                # A passive viewer collects mouse drags but never applies them itself.
                d.xfrc_applied[:] = 0
                mujoco.mjv_applyPerturbForce(m, d, pert)
                dragging = pert.active & mujoco.mjtPertBit.mjPERT_TRANSLATE
                root = roots.holds(pert.select, d) if dragging else None
                if root is not None:
                    d.xfrc_applied[pert.select, :3] = drag_force(m, d, pert)
                mujoco.mj_step(m, d)
                if root is not None:
                    pulling, pull = pert.select, roots.axial_force(d, root)
                for weed in roots.snap(d):
                    print(f"\nuprooted {weed} at {pull:.1f} N")

            if pulling is not None and frame_start >= next_readout:
                next_readout = frame_start + 0.1  # one self-overwriting line at 10 Hz
                print(f"  pulling {m.body(pulling).name} -- root feels {pull:5.1f} N "
                      f"of {ROOT_BREAK_FORCE:.0f} N", end="\r", flush=True)
            viewer.sync()
            time.sleep(max(0.0, substeps * m.opt.timestep - (time.time() - frame_start)))
    env.close()


def uproot(cfg):
    """Watch the arm do it: reach over the weed, take the stalk, twist, pull it out."""
    import mujoco.viewer
    from weed_extraction.uproot import Uprooter

    env = make_env(cfg, cameras=[])
    m, d = env.model, env.data
    runner = Uprooter(env, cfg.weed)
    print(f"{cfg.weed}: stalk at {np.round(runner.grip, 3)}, "
          f"root gives way at {ROOT_BREAK_FORCE:.0f} N of pull or "
          f"{ROOT_BREAK_TORQUE:.2f} N.m of twist, or any mix of the two")
    phase = None

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            name = runner.step()
            if name is None:
                break
            _, _, _, _, info = env.step(d.ctrl)
            if name != phase:
                phase = name
                print(f"  [{d.time:5.1f}s] {name}")
            for weed in info["uprooted"]:
                print(f"\n  uprooted {weed} at {runner.peak_pull:.1f} N of pull and "
                      f"{runner.peak_twist:.3f} N.m of twist\n")
            viewer.sync()
            time.sleep(max(0.0, runner.dt - (time.time() - frame_start)))
        if runner.uprooted is None:
            print(f"  {cfg.weed} stayed rooted: peak load {runner.peak_load:.2f} of 1.0")
        while viewer.is_running():  # leave the result on screen
            mujoco.mj_step(m, d, nstep=env.n_substeps)
            viewer.sync()
            time.sleep(runner.dt)
    env.close()


def capture(cfg):
    # MuJoCo's shadows are a hard in-or-out test with no filtering, and multisampling
    # cannot smooth them because shading is per pixel. Rendering large and shrinking
    # averages real pixels instead, which is what softens the stair-stepped edges.
    n = max(1, cfg.supersample)
    env = make_env(cfg, cfg.cameras, scale=n)
    mujoco.mj_step(env.model, env.data, nstep=round(cfg.settle_s / env.model.opt.timestep))
    obs = env.get_obs()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    size = (cfg.width, cfg.height)
    for cam in cfg.cameras:
        image = obs[cam]
        if image.ndim == 2:
            # Averaging depth across an edge would invent a surface that is not there.
            image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
            np.save(cfg.output_dir / f"{cam}.npy", image)  # metres
        else:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(cfg.output_dir / f"{cam}.png"), cv2.cvtColor(to_rgb(image), cv2.COLOR_RGB2BGR))
    print(f"saved {len(cfg.cameras)} camera(s) to {cfg.output_dir}")
    env.close()


def parse_config():
    """`weed-launch <config> [--key=value ...]`, where <config> is a path or a name in configs/."""
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit("usage: weed-launch <config> [--key=value ...]\nconfigs: "
                 f"{', '.join(p.stem for p in sorted(CONFIGS.glob('*.yaml')))}")
    config = Path(sys.argv[1])
    if not config.exists():
        config = CONFIGS / f"{sys.argv[1]}.yaml"
    return draccus.parse(LaunchConfig, config_path=config, args=sys.argv[2:])


def main():
    cfg = parse_config()

    # Absolute paths: on macOS the viewer changes the working directory.
    cfg.scene, cfg.output_dir = cfg.scene.resolve(), cfg.output_dir.resolve()

    if cfg.mode in ("view", "uproot"):
        # macOS only runs the passive viewer under mjpython; re-launch through it once.
        if sys.platform == "darwin" and not os.environ.get("WEED_UNDER_MJPYTHON"):
            os.environ["WEED_UNDER_MJPYTHON"] = "1"
            mjpython = str(Path(sys.executable).with_name("mjpython"))
            os.execv(mjpython, [mjpython, "-m", "weed_extraction.launch", *sys.argv[1:]])
        (view if cfg.mode == "view" else uproot)(cfg)
    elif cfg.mode == "capture":
        capture(cfg)
    else:
        sys.exit(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
