# CLAUDE.md

Living technical record for this repository. Maintained by Claude Code. Update this file after every completed implementation step (see **Maintenance Protocol** at the end).

---

## 0. Working Rules (standing defaults)

### 0.1 Be concise — overrides everything below

Answer only what was asked. Then stop.

- Lead with the answer. No preamble, no recap of the question, no closing summary.
- Plain English. No jargon, no function names or file:line unless asked where something lives.
- Default length is a few lines. A long answer needs a reason.
- No extras: no "what you should see", no how-to-verify, no caveats, no next steps, no recommendations — unless asked.
- Asked a yes/no question? Answer yes or no first.
- If something else genuinely needs saying, ask whether they want it.

This is a standing default, not a per-question request. It outranks any instruction below that would make an answer longer.

### 0.2 Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs. Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 0.3 Simplicity first

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked. No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it. Would a senior engineer call it overcomplicated? If yes, simplify.

### 0.4 Surgical changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting. Don't refactor what isn't broken. Match existing style.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that *your* changes made unused; leave pre-existing dead code unless asked.
- Test: every changed line traces directly to the user's request.

### 0.5 Goal-driven execution

Define success criteria. Loop until verified.

- "Add validation" → write tests for invalid inputs, then make them pass.
- "Fix the bug" → write a test that reproduces it, then make it pass.
- "Refactor X" → ensure tests pass before and after.
- Multi-step tasks: state a brief plan as `1. [Step] → verify: [check]`.
- After changes, summarize: files changed, what changed (1–2 bullets), how to test (exact commands).

### 0.6 README format

Every README uses this structure, concise throughout:

- **About** — 3 to 5 lines: what it is and why it exists. Include motivation, not just description.
- **Image** — a screenshot or diagram. If none exists, leave a placeholder with an HTML comment saying what should go there.
- **Build** — minimum necessary steps. Prerequisites in one line, then the commands.
- **Run** — how to run it, with one or two concrete examples using real paths/arguments.
- **Configuration** — options available at run time.
- **API** — optional, only if the thing is importable.

No tables, diagrams, badges, reference dumps, or "Contributing"/"License" boilerplate unless asked. Prose over bullets when explaining why; bullets for enumerating options. Drop any section that would be padding. The repository's README.md follows this.

## No Global Variables

**Module-level variables are the last resort, not the default.**

A constant belongs to whatever uses it:

* Used by one class -> class attribute. Constants stay class attributes, not `self.x = ...`
  in `__init__`, which implies they vary per instance.
* Used by one function -> local in that function.
* Used only as an argparse default -> write the literal in `add_argument`. Don't declare a
  constant and a function default for the same value; the parser always supplies it.

Module level only when it genuinely outweighs the alternatives - public API a caller imports,
or a value several classes in the file share - or when I ask for it. State the reason in a
comment when you do.

Applies to every module, not just new code: if you are editing a file, fix its globals.

---

## 1. Objective

**VLA-Weed-Extraction** — a MuJoCo simulation of an autonomous weeding robot for a raised-bed crop field.

A mobile manipulator (Franka Emika Panda on a rail) drives along a crop bed, recognises weeds from **visual and textural** cues rather than artificial marker colours, and pulls them out of the soil while leaving the crop untouched.

## 2. Final End-to-End Goal

One autonomous cycle, repeated for every weed:

1. Base moves along the crop bed.
2. Robot identifies a target weed from camera images (not colour codes).
3. Base stops at the weed.
4. Arm positions the gripper above it.
5. Gripper grasps the stem.
6. Arm pulls the weed out; root resistance is overcome.
7. Robot retreats and moves to the next target; crop plants are never grasped.

## 3. Scope and Architecture

**In scope:** MuJoCo simulation only — robot and field models, plant physics, perception on rendered images, base and arm control, grasp and extraction logic, an orchestrating state machine. Generating scripted demonstrations, packing them as a LeRobot dataset, and fine-tuning SmolVLA on them through the vendored `lerobot`.

**Out of scope for now:** real hardware, sim-to-real, training VLA models from scratch.

### Workspace layout (mandated)

Everything lives under `workspace/`, split like the team's ROS 2 workspaces so later ROS 2 packages drop into the same three folders. Third-party code is a git submodule inside the folder it belongs to. All XML asset paths are relative.

```
workspace/
├── hardware/      robot descriptions and drivers
│   ├── mujoco_menagerie/     submodule (Franka meshes)
│   ├── realsense-ros/        submodule (D435i reference)
│   └── panda_description/    mobile_panda.xml
├── application/   the world
│   └── field_description/    scene_mobile_field.xml, weeds.xml, assets/
└── planning/      motion API and launch
    ├── lerobot/              submodule (datasets, policies)
    └── weed_extraction/      the Python package
```

**Path rule** (verified on MuJoCo 3.8.1, and still holding on 3.13.0, the pinned version): `include` files, and any `meshdir` set inside one, resolve relative to the **top-level scene file**. So `mobile_panda.xml`'s `meshdir` is written relative to `workspace/application/field_description/`, and mesh paths in `weeds.xml` are written relative to that inherited `meshdir` (four levels up). Textures resolve relative to the scene file instead. The scene compiles from any working directory.

## 4. Current Repository Structure

```
VLA-Weed-Extraction/
├── README.md                 human-facing: setup, dataset building, SmolVLA training setup
├── CLAUDE.md                 this file
├── Dockerfile                Ubuntu 24.04 + ROS 2 Jazzy, MuJoCo; LeRobot with --build-arg WITH_SMOLVLA=true
├── docs/
│   ├── scene.png                        README screenshot
│   ├── vla_training_automation_plan.md  the SmolVLA plan (single source of truth)
│   └── smolvla_data_collection.md       retired, points to the plan
├── run_scripts/
│   ├── setup.sh              native venv setup; --smolvla adds LeRobot and the smolvla extra
│   ├── simulate.sh           launcher; works natively and inside the container
│   ├── test_up_root.sh       scripted grasp-twist-pull demo in the viewer
│   ├── train.sh              fine-tune SmolVLA on dataset v2 (bed and wrist cameras, vision frozen)
│   ├── demo.sh               type a prompt: scan the bed, park at a matching weed, pick it
│   └── ui.sh                 browser UI: pick a held-out environment, type a prompt, the demo runs in the native viewer
├── docker_scripts/run_scripts/   build (passes --build-arg), base, devel, up, enter_bash, stop
├── models/                   Git LFS: detector.pt (the demo's weed detector), smolvla_weeds/ (the baseline SmolVLA, 20,000 steps)
├── outputs/                  generated scenes and datasets; git-ignored
└── workspace/
    ├── hardware/
    │   ├── mujoco_menagerie/         submodule @ 71f066a
    │   ├── realsense-ros/            submodule @ 9a11121 (tag 4.58.4)
    │   └── panda_description/mobile_panda.xml
    ├── application/field_description/
    │   ├── scene_mobile_field.xml    ACTIVE SCENE
    │   ├── weeds.xml                 the 8 weeds, included by the scene
    │   └── assets/
    │       ├── ground/               grass + soil textures (Poly Haven, CC0); variants/ for sampled scenes
    │       └── plants/{strawberry,sowthistle,shepherd}/
    │           ├── *_plant.png       texture
    │           ├── bands/            visual mesh, one per stalk segment
    │           └── collision/        CoACD convex pieces
    └── planning/
        ├── lerobot/                  submodule @ bad0260 (v0.6.1)
        └── weed_extraction/
            ├── pyproject.toml        pinned deps; extras: dev, smolvla
            ├── weed_extraction/
            │   ├── env.py            Gymnasium env, breakable roots, explicit grasp weld
            │   ├── ik.py             collision-aware arm IK (mink), built per scene
            │   ├── uproot.py         scripted expert: reach, grip, twist, lift, carry, drop
            │   ├── launch.py         weed-launch: view, capture, uproot; --seed for sampled scenes
            │   ├── scene.py          manifests, six scene families, placement rules, MJCF templating
            │   ├── prompts.py        language prompts and the weeds each one names
            │   ├── recovery.py       bounded grasp and extraction retries, attempt labels
            │   ├── recorder.py       causal episode recording to observations.npz + episode.json
            │   ├── validate.py       episode acceptance checks and rejection reasons
            │   ├── generate.py       planning and process-isolated parallel generation
            │   ├── pack.py           LeRobot Dataset v3 packing and inspection
            │   ├── dataset.py        chunked build: generate, pack, check, prune, merge
            │   ├── policy.py         SmolVLA input boundary: depth step, features, model directory
            │   ├── detector.py       weed detector on the bed camera: render labelled pictures, train, place plants
            │   ├── demo.py           state machine: back up, scan, go, pick (the scripted expert), return home; asks for the next prompt
            │   ├── smolvla_probe.py  one SmolVLA inference on a live observation; prints the 9-D action, never applies it
            │   ├── vla_runner.py     SmolVLA in the loop: observation in, one 9-D action per tick applied with env.step (not used by the UI)
            │   └── ui.py             browser UI (stdlib http.server) around demo.Demo; one worker thread owns the scene and the viewer
            ├── configs/{view,capture,uproot}.yaml
            └── tests/                16 files: test_scene (the MJCF), test_manifest (scene.py), test_ui (ui.py, vla_runner.py), one per other module but launch.py
```

### Active scene: `workspace/application/field_description/scene_mobile_field.xml`

Includes `mobile_panda.xml` (robot) and `weeds.xml` (weeds). Owns the beds, the walking path, lighting, the 8 strawberries and the overview camera.

Compiled size: `nq=114 nv=106 nu=9 nbody=55 ngeom=389 nmesh=146 neq=17 ncam=3`.

Cameras (3): `rgb_camera` — worldbody, `mode="track" target="link0"`, third-person overview. `d435i_color` and `d435i_depth` — the RealSense D435i on the wrist, declared inside `mobile_panda.xml`.

## 5. Dataset and Policy Pipeline

`python -m weed_extraction.dataset <out_dir>` runs the whole pipeline, in chunks that fit the disk:

1. **Plan**: 4 episodes in each of the 20 training scenes and 1 in each of the 10 held-out scenes, 90 in all. Each episode is one single-target prompt, balanced over the prompt kinds each scene offers.
2. **Generate**: one worker process per episode. The recorder reads the state before each action, then the expert (with recovery) acts. Validation then accepts or rejects the episode with a reason.
3. **Pack**: each chunk is packed to LeRobot v3 and inspected against its raw frames. The raw frames, accepted and rejected, are then deleted; they can be regenerated deterministically.
4. **Merge**: one dataset per split, checked against its chunks.

Splits: `held_out_family` is all of evening_cool. `held_out_variant` is variant 4 of the other five families. `train` is everything else.

Current build (`outputs/dataset_v1`): 70 / 5 / 4 accepted episodes (68,800 / 4,750 / 3,800 frames). 11 were rejected: 10 for crop contact, 1 for a tray drop.

Policy interface:
- `observation.state` has 28 values: 10 joint positions, 10 velocities, a 7-value end-effector pose and the gripper width.
- Images: wrist RGB, wrist depth and overview RGB, each 144×256.
- The language prompt.
- `action` has 9 targets: 7 joints, the gripper and the rail.

The depth is stored in metres. `policy.DepthToRGBProcessorStep` heads SmolVLA's preprocessor and converts it: over 2 m, clipped, into 3 channels. Its settings are saved with the preprocessor.

`label.*` (root wrench, phase, attempt, held, uprooted) is never a policy input.

`python -m weed_extraction.policy` writes the `lerobot/smolvla_base` model directory; `--cameras bed wrist` limits the policy to those images, and the depth step is then left out. `run_scripts/train.sh` does both steps for dataset v2. By hand, train with `lerobot_train.py --policy.path=<dir> --env.discover_packages_path=weed_extraction`.

### Running the demo

`run_scripts/demo.sh "remove the shepherd weed"` runs a state machine in `demo.py`, with no trained policy. The base backs to the start of the rail, drives along the bed at 0.3 m/s while the weed detector maps the bed camera's view, and parks at the first matching weed. The scripted expert (`Uprooter`, as in `test_up_root.sh`) then picks the weed nearest the detector's placement and drops it in the bin. If the weed is not in the bin, the arm returns to the home pose and the expert tries again, up to 3 picks. After returning home the demo asks for the next prompt in the same scene; Enter quits. Roots are at half strength in the demo only.

The trained models were dropped from the demo: the baseline and the pick-only model (`outputs/train/smolvla_pick`, 10,000 steps at batch 16) both closed the jaws 2 to 3 cm from the stem.

The weed detector (`detector.py`) is a torchvision Faster R-CNN over three labels: sowthistle, shepherd, strawberry. `render` saves bed camera pictures from stops along the rail of 54 catalogue scenes, boxed by the simulator's segmentation render. `train` fits the 40 training scenes and checks the 14 held-out ones. A box is placed on the bed where its ray meets a plane at a height measured per label.

### Browser UI

`run_scripts/ui.sh` serves `http://127.0.0.1:8765/` and opens it in the browser. The flow is: pick an environment from its picture, type a prompt (autocomplete offers plain-text suggestions), EXECUTE. Only then is the scene built (or reused) and the native MuJoCo viewer opened, as with `demo.sh`; the page itself shows only the wrist camera, the live 9-D action and the status. The robot is run by the demo's own state machine (`demo.Demo`, picking with `Uprooter`) through `demo.run`, exactly as `demo.sh` runs a prompt; no trained policy is used. Flags: `--detector` (default `models/detector.pt`), `--seconds` (as the demo: 90 simulated seconds, 600 for an "all" prompt), `--port`, `--no-browser`, `--no-viewer` (no display, for tests), `--log-first/--log-last/--log-every`. Only the Python standard library and Pillow: the project's venv has no GUI toolkit.

- **Threads**: request handlers read a lock-protected snapshot (wrist image, applied action, step, time, status) and put `execute` commands on a queue; a second command while one runs is refused (409). STOP is a `threading.Event`, cleared when a prompt is accepted, so a STOP pressed while the run is still loading counts. One worker thread owns the demo, its scene and the viewer. `demo.run` is handed `ui.Live` as its viewer: after every tick it syncs the native viewer and renders the wrist camera off-screen from the same `MjData`, and it reports the run over on STOP or a closed window.
- **Rendering setup**: importing `weed_extraction` already loads mujoco and glfw, so `ui.main()` re-executes itself once with `MUJOCO_GL=egl` (the wrist and bed renderers never touch GLFW) and, when there is an X display, `PYGLFW_LIBRARY_VARIANT=x11` (the viewer on XWayland). With GLFW's Wayland build, or with GLFW rendering from the worker thread, a closed viewer's render loop could stay blocked and hang GLFW's cleanup at exit. Closing a viewer also waits for its render thread to end.
- **Environments**: only the held-out ones, from `Manifest._split` over the 9-variant catalogue: evening_cool 0 to 8 and variant 4 of the other five families (14). Each picture is that scene's own overview camera with the robot at home, rendered once into `outputs/ui/thumbs/` by child processes at start-up. A scene is built by `Demo` itself (bed camera, 25 Hz, the demo's halved roots).
- **Lifecycle**: a `Demo` (scene and detector) is built when an environment is first used and kept while it stays selected. Every prompt resets the scene and re-applies the demo's root easing, so each starts as a new `Demo` would. Choosing another environment closes the viewer, the wrist renderer and the scene before building the next (each compiled scene still holds about 200 MB until the process exits). Closing the viewer window stops the run ("Viewer closed."); the next prompt opens a new one. Ctrl+C or SIGTERM stops the server, then the worker closes the viewer and the scene.
- **Prompt check** (`ui.PromptCheck`): anything may be typed. A prompt reaches the demo only if `demo.Command.parse` accepts it (one weed type, at most one bed, optionally all/every/field/standing), it names no strawberry and has no negation (not, don't, never, except, without, any n't). It is passed to the demo as typed. Anything else shows "Invalid prompt. Please provide a valid weed-extraction prompt." and no scene is built or touched.
- **Autocomplete**: the 24 trained wordings (both weed types, alone and with each bed), matched word by word while typing, shown as plain text.
- **Ends**: the state machine back home → FINISHED (the terminal prints the demo's RESULT text); STOP, the time limit or a closed viewer → STOPPED; any exception → ERROR.
- **Telemetry**: the page shows `env.data.ctrl` after each tick. The demo's `ActionLog` logs the same values to `outputs/demo_logs/<time>/`.
- **Terminal**: the demo's own printout (states, map, actions, summary) is redirected away during a UI run; the terminal shows only the server URL, environment loaded, execution started, and completed (with the demo's RESULT text) or stopped. Tracebacks still go to stderr. `demo.sh` prints as before.
- Checked 2026-10-11: through the UI worker, low_light 4, "remove the shepherd weed from the left bed" binned shepherd_1 and finished in 28.9 simulated seconds.

## 6. Maintenance Protocol

After every completed implementation step, update this file: structure, pipeline, and the state below. Remove claims that stopped being true rather than appending corrections.

**State (2026-10-03):**
- 2026-10-11: 310 tests, 302 passed in 31 minutes (the slow UI tests open the native viewer and need a display; the UI tests passed again after the UI moved to the demo). The 8 errors are `test_pack.py`'s module fixture: it expects both of its episodes accepted, but the held-out one (evening_cool 0, "remove the sowthistle weed") is rejected for crop contact (mobile base on strawberry_8, 413 N). Tests needing the `smolvla` extra skip without it.
- Pinned: MuJoCo 3.13.0, mink 1.3.0; the `smolvla` extra pins torch 2.11.0, transformers 5.5.4 and datasets 4.8.5.
- The baseline (dataset v2, 20,000 steps at batch 5, `outputs/train/smolvla_weeds`) does not reach a weed: it guesses where to drive, because most recorded weeds start out of the bed camera's view. `docs/adr/0002` splits the task instead.
- Weed detector on the 14 held-out scenes: 382 of 383 sowthistles and 322 of 347 shepherd weeds found, no wrong types, median 5 to 6 mm off along the bed.
- The demo with the baseline as pick model: parked at a weed in 8 of 8 runs, binned it in 1.
- `python -m weed_extraction.dataset <dir> --parked` records pick-only episodes: the base is parked beside the weed before recording, and every prompt names type and bed. `run_scripts/train.sh` trains on `outputs/dataset_pick`, which was being built on 2026-10-03 (137 episodes planned); the pick model is not trained yet.
- On the RTX 4060 Laptop (8 GB): vision frozen fits at batch 16 (4.9 GB, about 1.3 s a step). Training the vision and language parts together runs out of memory at batch 4 and 8; training only part of them is untested. torchcodec cannot load there, so `test_lerobot_train_consumes_the_model_directory` fails and training reads videos with pyav.

**Known deviations from the plan:**
- 50 Hz recording, not 20 fps.
- Single-target prompts only.
- Tray-drop failures are detected, not retried.
- Crop safety is measured as contact, and the expert cannot see crops.
- Target occlusion is not measured.
- The root wrench is a label, not a wrist force/torque sensor.
- Held-out evaluation has not been run.
