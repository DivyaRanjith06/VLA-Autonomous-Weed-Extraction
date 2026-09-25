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
│   └── test_up_root.sh       scripted grasp-twist-pull demo in the viewer
├── docker_scripts/run_scripts/   build (passes --build-arg), base, devel, up, enter_bash, stop
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
            │   └── policy.py         SmolVLA input boundary: depth step, features, model directory
            ├── configs/{view,capture,uproot}.yaml
            └── tests/                13 files: test_scene (the MJCF), test_manifest (scene.py), one per other module but launch.py
```

### Active scene: `workspace/application/field_description/scene_mobile_field.xml`

Includes `mobile_panda.xml` (robot) and `weeds.xml` (weeds). Owns the beds, the walking path, lighting, the 8 strawberries and the overview camera.

Compiled size: `nq=114 nv=106 nu=9 nbody=55 ngeom=389 nmesh=146 neq=17 ncam=3`.

Cameras (3): `rgb_camera` — worldbody, `mode="track" target="link0"`, third-person overview. `d435i_color` and `d435i_depth` — the RealSense D435i on the wrist, declared inside `mobile_panda.xml`.

## 5. Dataset and Policy Pipeline

`python -m weed_extraction.dataset <out_dir>` runs the whole pipeline, in chunks that fit the disk:

1. **Plan**: 4 episodes in each of the 20 training scenes and 1 in each of the 10 held-out scenes, 90 in all. Each episode is one single-target prompt, balanced over the prompt kinds each scene offers.
2. **Generate**: one worker process per episode. The recorder reads the state before each action, then the expert (with recovery) acts. Validation then accepts or rejects the episode with a reason.
3. **Pack**: each chunk is packed to LeRobot v3 and inspected against its raw frames. The accepted raw frames are then deleted; they can be regenerated deterministically.
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

`python -m weed_extraction.policy` writes the `lerobot/smolvla_base` model directory. Train with `lerobot_train.py --policy.path=<dir> --env.discover_packages_path=weed_extraction`.

## 6. Maintenance Protocol

After every completed implementation step, update this file: structure, pipeline, and the state below. Remove claims that stopped being true rather than appending corrections.

**State (2026-09-25):**
- 230 tests pass, about 20 minutes. Tests needing the `smolvla` extra skip without it.
- Pinned: MuJoCo 3.13.0, mink 1.3.0; the `smolvla` extra pins torch 2.11.0, transformers 5.5.4 and datasets 4.8.5.
- The pre-training dataset is built and verified. SmolVLA training has not been run beyond a few smoke-test steps.

**Known deviations from the plan:**
- 50 Hz recording, not 20 fps.
- Single-target prompts only.
- Tray-drop failures are detected, not retried.
- Crop safety is measured as contact, and the expert cannot see crops.
- Target occlusion is not measured.
- The root wrench is a label, not a wrist force/torque sensor.
- Held-out evaluation has not been run.
