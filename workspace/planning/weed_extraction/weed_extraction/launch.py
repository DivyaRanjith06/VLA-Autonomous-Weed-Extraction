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

from weed_extraction.env import ROOT_BREAK_FORCE, SCENE, WeedExtractionEnv

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# The viewer scales its own drag force by the dragged body's mass, so a 15 g weed
# feels ~0.3 N against a root that holds at 15 N -- you could never pull one out.
# While the root holds, drag the weed with this spring instead: DRAG_TRAVEL of
# cursor movement produces exactly the force the root gives way at, wherever on the
# plant you grabbed it. The spring is critically damped against the grabbed body's
# mass, without which a 2 g stalk tip rings at ~80 Hz and throws force spikes into
# the root, and clamped so a fast flick cannot either. Once the weed is loose the
# viewer's own force takes over, or it would shoot off to the cursor.
DRAG_TRAVEL = 0.025  # metres of drag that produces ROOT_BREAK_FORCE
DRAG_FORCE_CAP = 1.3 * ROOT_BREAK_FORCE  # a flick cannot hugely overshoot the root


def drag_force(model, data, pert, stiffness):
    """The pull a mouse drag puts on a weed: a critically damped spring to the cursor."""
    body = pert.select
    stretch = pert.refselpos - data.xipos[body]
    damping = 2 * np.sqrt(stiffness * model.body_mass[body])  # critical: no ringing
    force = stiffness * stretch - damping * data.cvel[body, 3:]
    size = np.linalg.norm(force)
    return force * (DRAG_FORCE_CAP / size) if size > DRAG_FORCE_CAP else force


@dataclass
class LaunchConfig:
    mode: str = "view"  # view: interactive viewer; capture: save one frame per camera
    scene: Path = SCENE
    cameras: list[str] = field(default_factory=lambda: ["d435i_color", "d435i_depth"])  # capture only
    width: int = 640
    height: int = 360
    base_x: float = 0.0  # base target along the bed, metres
    settle_s: float = 2.0  # capture: simulated seconds before the frame is taken
    supersample: int = 2  # capture: render this many times larger, then shrink
    output_dir: Path = Path("outputs")


def make_env(cfg, cameras):
    env = WeedExtractionEnv(cfg.scene, cameras, cfg.width, cfg.height)
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
    stiffness = ROOT_BREAK_FORCE / DRAG_TRAVEL
    # One frame is many physics steps: syncing the viewer every 1 ms step is far
    # slower than the physics itself, and the window would crawl behind real time.
    substeps = max(1, round(1 / 60 / m.opt.timestep))
    next_readout = 0.0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            pulling, pull = None, 0.0
            for _ in range(substeps):
                # A passive viewer collects mouse drags but never applies them itself.
                mujoco.mjv_applyPerturbForce(m, d, viewer.perturb)
                pert = viewer.perturb
                if (pert.active & mujoco.mjtPertBit.mjPERT_TRANSLATE
                        and roots.holds(pert.select, d) is not None):
                    d.xfrc_applied[pert.select, :3] = drag_force(m, d, pert, stiffness)
                    pulling = pert.select
                mujoco.mj_step(m, d)
                d.xfrc_applied[:] = 0
                if pulling is not None:
                    pull = roots.axial_force(d, roots.of_body[pulling])
                for weed in roots.snap(d):
                    print(f"\nuprooted {weed} at {pull:.1f} N")

            if pulling is not None and frame_start >= next_readout:
                next_readout = frame_start + 0.1  # one self-overwriting line at 10 Hz
                print(f"  pulling {m.body(pulling).name} -- root feels {pull:5.1f} N "
                      f"of {ROOT_BREAK_FORCE:.0f} N", end="\r", flush=True)
            viewer.sync()
            time.sleep(max(0.0, substeps * m.opt.timestep - (time.time() - frame_start)))
    env.close()


def capture(cfg):
    # MuJoCo's shadows are a hard in-or-out test with no filtering, and multisampling
    # cannot smooth them because shading is per pixel. Rendering large and shrinking
    # averages real pixels instead, which is what softens the stair-stepped edges.
    n = max(1, cfg.supersample)
    env = WeedExtractionEnv(cfg.scene, cfg.cameras, cfg.width * n, cfg.height * n)
    env.reset()
    env.data.ctrl[env.model.actuator("base_x").id] = cfg.base_x
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


def parse_config(cls, command):
    """`<command> <config> [--key=value ...]`, where <config> is a path or a name in configs/."""
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        sys.exit(f"usage: {command} <config> [--key=value ...]\nconfigs: "
                 f"{', '.join(p.stem for p in sorted(CONFIGS.glob('*.yaml')))}")
    config = Path(sys.argv[1])
    if not config.exists():
        config = CONFIGS / f"{sys.argv[1]}.yaml"
    return draccus.parse(cls, config_path=config, args=sys.argv[2:])


def main():
    cfg = parse_config(LaunchConfig, "weed-launch")

    # Absolute paths: on macOS the viewer changes the working directory.
    cfg.scene, cfg.output_dir = cfg.scene.resolve(), cfg.output_dir.resolve()

    if cfg.mode == "view":
        # macOS only runs the passive viewer under mjpython; re-launch through it once.
        if sys.platform == "darwin" and not os.environ.get("WEED_UNDER_MJPYTHON"):
            os.environ["WEED_UNDER_MJPYTHON"] = "1"
            mjpython = str(Path(sys.executable).with_name("mjpython"))
            os.execv(mjpython, [mjpython, "-m", "weed_extraction.launch", *sys.argv[1:]])
        view(cfg)
    elif cfg.mode == "capture":
        capture(cfg)
    else:
        sys.exit(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
