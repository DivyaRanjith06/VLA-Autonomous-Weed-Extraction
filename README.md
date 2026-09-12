# VLA-Weed-Extraction

A MuJoCo simulation of a weeding robot: a Franka Panda arm on a rail drives along two
raised beds planted with strawberries and weeds.

It exists so the weeding behaviour can be developed without a field. The weeds are
modelled properly — stalks that bend and spring back, roots that tear only under a firm
upward pull — so anything that works here had to solve the real problem rather than a
toy version of it.

![The field scene](docs/scene.png)

## Contents

- [Requirements](#requirements)
- [Run it natively](#run-it-natively)
- [Run it in Docker](#run-it-in-docker)
- [Pull a weed out](#pull-a-weed-out)
- [Configuration](#configuration)
- [Tests](#tests)
- [API](#api)

## Requirements

Pick one path. You do not need both.

**Natively** — macOS, Linux or Windows (WSL):

- [git](https://git-scm.com/downloads)
- [Python 3.12 or newer](https://www.python.org/downloads/) — Ubuntu:
  `sudo apt install git python3.12 python3.12-venv`, macOS: `brew install git python@3.12`

Everything else installs into a virtualenv inside the repo. Nothing touches your system
Python.

**In Docker** — Linux:

- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) (on macOS or Windows,
  [Docker Desktop](https://docs.docker.com/get-started/get-docker/) — but the viewer
  needs an X display, so run natively there instead)

## Run it natively

Fastest way to see it. Works on macOS, Linux and Windows (WSL).

### First-time setup

Check your Python is new enough, then clone and install:

```
python3 --version    # 3.12 or newer
```

```
git clone --recurse-submodules https://github.com/DivyaRanjith06/VLA-Weed-Extraction.git
cd VLA-Weed-Extraction
python3.12 -m venv .venv
.venv/bin/pip install -e "workspace/planning/weed_extraction[dev]"
```

That pulls MuJoCo, Gymnasium, NumPy, OpenCV and draccus at pinned versions. Nothing is
installed outside `.venv`.

Cloned without `--recurse-submodules`? The Panda meshes are missing and nothing will
load. Fix it with:

```
git submodule update --init workspace/hardware/mujoco_menagerie
```

### Every time after that

```
./run_scripts/simulate.sh
```

## Run it in Docker

For Linux. Use this if you would rather not install Python 3.12, or you want the ROS 2
tools that are already in the image.

### First-time setup

```
git clone --recurse-submodules https://github.com/DivyaRanjith06/VLA-Weed-Extraction.git
cd VLA-Weed-Extraction/docker_scripts/run_scripts
./build.sh
```

Once, and the build takes a while. The image carries its own copy of the workspace, so
nothing else has to be installed.

### Every time after that

```
cd docker_scripts/run_scripts
./up.sh             # start the container and launch in one go
./stop.sh           # when you are done
```

The MuJoCo window opens on your desktop, exactly as it does natively.

To work inside the container instead:

```
./devel.sh          # start it, with the repo mounted
./enter_bash.sh     # get a shell
./run_scripts/simulate.sh
```

`devel.sh` mounts the repo over the copy baked into the image, so your edits take
effect with no rebuild. `base.sh` starts it without the mount, running the image's own
copy.

On a server with no display, everything still renders offscreen:
`MUJOCO_GL=egl weed-launch capture`

## Pull a weed out

Once the window is open:

- Scroll to zoom in on one of the green plants with no berries.
- Double-click it to select.
- Hold **Ctrl** and **right-drag upward**, about 3 cm.
- It stretches, tears loose and drops onto the soil. The terminal prints the force.

Drag sideways instead and it only bends — the root gives way to an upward pull, the way
a real one does. Press **F1** for the viewer's other controls.

## Configuration

`simulate.sh` passes everything through to the launcher.

- `./run_scripts/simulate.sh` — the viewer (default)
- `./run_scripts/simulate.sh capture` — save one frame per camera into `outputs/`
- `./run_scripts/simulate.sh capture --supersample=4` — slower, but the cleanest edges
- `./run_scripts/simulate.sh view --base_x=1.5` — start with the arm 1.5 m along the bed

Presets live in `workspace/planning/weed_extraction/configs/`. Any field in one can be
overridden with `--name=value`.

Three cameras: `rgb_camera` follows the robot from a distance, `d435i_color` and
`d435i_depth` are the RealSense D435i on the wrist.

## Tests

```
cd workspace/planning/weed_extraction
../../../.venv/bin/python -m pytest
```

Fourteen checks covering the scene, the gripper's grip on a stem, the root breaking at
the right force and only in the right direction, the stalk bending and springing back,
and the crop staying put.

## API

The simulation is a Gymnasium environment:

```python
import gymnasium as gym
import weed_extraction  # registers the env

env = gym.make("WeedExtraction-v0", cameras=["d435i_color"])
obs, _ = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["uprooted"])   # names of any weeds torn loose this step
```

Actions are the nine actuator targets: seven arm joints, the gripper (0 closed to 255
open) and the base position in metres. Observations are joint positions, velocities and
one image per camera. Reward is 0 — no task is defined yet.
