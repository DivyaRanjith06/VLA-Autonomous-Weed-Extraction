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
- [Watch the robot do it](#watch-the-robot-do-it)
- [Build a training dataset](#build-a-training-dataset)
- [Prepare SmolVLA training](#prepare-smolvla-training)
- [Current status](#current-status)
- [Configuration](#configuration)
- [Tests](#tests)
- [API](#api)

## Requirements

Pick one path. You do not need both.

**Natively** — macOS, Linux or Windows (WSL):

- [git](https://git-scm.com/downloads)
- [Python 3.12 or newer](https://www.python.org/downloads/) — Ubuntu 22.04:
  install it from the [Deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa),
  then install `python3.12-venv`; macOS: `brew install git python@3.12`

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

Building a dataset or training SmolVLA needs LeRobot and PyTorch as well, as the
`smolvla` extra, pinned like the rest. LeRobot is the submodule, installed without its
own dependency list, since the extra already pins what it needs:

```
git submodule update --init workspace/planning/lerobot
.venv/bin/pip install -e "workspace/planning/weed_extraction[dev,smolvla]"
.venv/bin/pip install --no-deps -e workspace/planning/lerobot
```

`./run_scripts/setup.sh` does the plain setup in one step, and
`./run_scripts/setup.sh --smolvla` the setup with the extra.

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
nothing else has to be installed. For dataset building and SmolVLA training, build with
`./build.sh --build-arg WITH_SMOLVLA=true`.

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

## Watch the robot do it

```
./run_scripts/test_up_root.sh
```

The arm drives to a weed, comes in on the stalk from the side rather than straight down
its axis, closes on it 16 mm above the soil, gives it a short turn about the vertical and
lifts. The terminal names each phase and prints the pull and the twist the root felt when
it gave way.

Roots fail on twist as well as pull, so the two are combined: 15 N of straight pull will
do it, so will 0.07 N·m of twist on its own, and any mix in between. That is why twisting
works — it buys you the pull the arm would otherwise have to apply.

Pick a different weed with `--weed`, for example
`./run_scripts/test_up_root.sh --weed=sowthistle_2`. Current status says what to expect
from each species.

## Build a training dataset

Needs the `smolvla` extra.

```
.venv/bin/python -m weed_extraction.dataset outputs/dataset_v1
```

This records 90 demonstrations across the 30 catalogue scenes: six scene families
(midday dry and loose soil, evening warm and cool, low light, mixed clutter), five
variants each. Every scene is drawn from a fixed seed, so the same command rebuilds the
same dataset. Four episodes go to each training scene and one to each held-out scene, with
the language prompts spread over the kinds each scene can answer. Every prompt names a
single weed.

Each episode is the scripted expert at 50 Hz, with up to three attempts if a grasp or
extraction fails, followed by validation. Rejected episodes are kept with their reasons in
`rejected.jsonl`. Accepted ones are packed into LeRobot Dataset v3, one dataset per split:

- `outputs/dataset_v1/lerobot/train` — the four known variants of five families
- `outputs/dataset_v1/lerobot/held_out_family` — every variant of evening cool, never
  seen in training
- `outputs/dataset_v1/lerobot/held_out_variant` — the fifth variant of each other family

The current build has 70 training episodes (68,800 frames), 5 held-out-family (4,750) and
4 held-out-variant (3,800); 11 of the 90 were rejected, 10 of them for touching a
strawberry.

A raw episode is about 180 MB, so the batch is built in chunks of 15. Each chunk is
packed and checked against its raw frames, and then those frames are deleted; everything
else is kept, and any episode can be regenerated exactly. Allow about 6 GB of free disk;
rejected episodes are kept whole, about 2 GB of that. The current build took about three
hours with two worker processes on a 16 GB machine, and an interrupted build carries on
where it stopped. `--train-per-scene`,
`--held-out-per-scene`, `--chunk-size` and `--workers` change the defaults.

What the policy sees:

- `observation.state` — 28 values: the ten joint positions (rail, seven arm joints, two
  fingers), their ten velocities, the gripper's pose (position and quaternion) and the
  gripper width
- three camera images, 144×256: wrist RGB, wrist depth and the overview camera
- the language prompt, for example "remove the shepherd weed"
- `action` — nine targets: seven arm joints, the gripper (0–255) and the rail position

The wrist depth is stored as the camera measured it, in metres. SmolVLA's vision encoder
takes three-channel images, so a preprocessing step at the policy's input converts it:
depth over 2 m, clipped to 0–1, the same in all three channels. The step is saved with the
policy's preprocessor, so inference converts depth exactly as training did.

Everything under `label.*` is for evaluation, never an input. That includes the root
wrench, the simulator's true pull and twist at the root, which no sensor on the robot
measures.

## Prepare SmolVLA training

Fine-tuning starts from `lerobot/smolvla_base` (0.9 GB, downloaded on first use). First
write a model directory fitted to the dataset. It holds the checkpoint's weights as a link,
its config set to the dataset's features, and the preprocessing above:

```
.venv/bin/python -m weed_extraction.policy outputs/dataset_v1/lerobot/train outputs/smolvla_weeds
```

Then train with LeRobot's own script:

```
.venv/bin/python -m lerobot.scripts.lerobot_train \
    --policy.path=outputs/smolvla_weeds \
    --env.discover_packages_path=weed_extraction \
    --dataset.repo_id=weed_extraction/dataset_v1_train \
    --dataset.root=outputs/dataset_v1/lerobot/train \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/smolvla_weeds
```

`--env.discover_packages_path=weed_extraction` loads this package so LeRobot knows the
depth step. It must not be written as `--policy.discover_packages_path`, which LeRobot
rejects. So far training has only been checked to run for a few steps. On a CPU a step
takes about half a minute, so real training needs a GPU or Apple silicon.

## Current status

The default weed, `shepherd_1`, comes out the way it is meant to: the root gives way during
the lift, on pull and twist together.

All four sowthistles now reach the grip without being knocked over, register as held, and
are carried out of the bed. Their roots break too early, though. The twist alone breaks
them, at about 0.4–0.5 N of pull, so the lift carries a plant that is already loose. Treat
sowthistle extraction as work in progress, not as a realistic pull-and-twist extraction.

Two things make the sowthistle grasp work:

- **The hand and wrist camera pass through sowthistle leaves.** Real leaves are thin and
  get brushed aside; here they are rigid, and the wide sowthistle canopy otherwise gets
  shoved over before the fingers arrive.
- **The grasp has a 1 mm tolerance.** A grasp counts when both fingertips touch the stem,
  or when one fingertip is on the stem and the other is within 1 mm of it. The two
  fingers move together, so if the arm stops a fraction of a millimetre off-centre, the
  first finger to touch stops both. Only contact with the target plant's own stem counts,
  and only once the gripper has been told to close.

The test suite passes: 230 tests.

## Configuration

`simulate.sh` passes everything through to the launcher.

- `./run_scripts/simulate.sh` — the viewer (default)
- `./run_scripts/simulate.sh capture` — save one frame per camera into `outputs/`
- `./run_scripts/simulate.sh capture --supersample=4` — slower, but the cleanest edges
- `./run_scripts/simulate.sh view --base_x=1.5` — start with the arm 1.5 m along the bed
- `./run_scripts/simulate.sh view --seed=5 --family=evening_warm` — a sampled scene
  instead of the hand-written one, written to `outputs/scenes/`; the same seed always
  gives the same scene, and `--family` is drawn from the seed when left out

Presets live in `workspace/planning/weed_extraction/configs/`. Any field in one can be
overridden with `--name=value`.

Three cameras: `rgb_camera` follows the robot from a distance, `d435i_color` and
`d435i_depth` are the RealSense D435i on the wrist.

## Tests

```
cd workspace/planning/weed_extraction
../../../.venv/bin/python -m pytest
```

230 checks. They cover:

- the scene, the gripper's grip on a stem, the root breaking at the right force and only
  in the right direction, the stalk bending and springing back, the IK keeping the arm out
  of the beds, and extraction end to end;
- scene sampling and its placement rules, the prompts and the recovery attempts;
- the recorder's causal alignment, episode validation, and parallel generation;
- packing to LeRobot, the chunked dataset build, the SmolVLA interface and the model
  directory.

The full run takes about 20 minutes, most of it simulating real episodes. Without the
`smolvla` extra, the tests that need it are skipped.

If you have ROS sourced and pytest fails before running anything with
`No module named 'lark'`, run it as
`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../../.venv/bin/python -m pytest`.

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
