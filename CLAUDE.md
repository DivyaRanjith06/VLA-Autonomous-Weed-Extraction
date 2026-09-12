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

**In scope:** MuJoCo simulation only — robot and field models, plant physics, perception on rendered images, base and arm control, grasp and extraction logic, an orchestrating state machine. Optional episode logging via the vendored `lerobot`.

**Out of scope for now:** real hardware, sim-to-real, training large VLA models.

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

**Path rule** (verified, MuJoCo 3.8.1): `include` files, and any `meshdir` set inside one, resolve relative to the **top-level scene file**. So `mobile_panda.xml`'s `meshdir` is written relative to `workspace/application/field_description/`, and mesh paths in `weeds.xml` are written relative to that inherited `meshdir` (four levels up). Textures resolve relative to the scene file instead. The scene compiles from any working directory.

## 4. Current Repository Structure

```
VLA-Weed-Extraction/
├── README.md                 human-facing: native setup, then Docker
├── CLAUDE.md                 this file
├── Dockerfile                Ubuntu 24.04 + ROS 2 Jazzy, MuJoCo + LeRobot
├── docs/scene.png            README screenshot
├── run_scripts/simulate.sh   launcher; works natively and inside the container
├── docker_scripts/run_scripts/   build, base, devel, up, enter_bash, stop
└── workspace/
    ├── hardware/
    │   ├── mujoco_menagerie/         submodule @ 71f066a
    │   ├── realsense-ros/            submodule @ 9a11121 (tag 4.58.4)
    │   └── panda_description/mobile_panda.xml
    ├── application/field_description/
    │   ├── scene_mobile_field.xml    ACTIVE SCENE
    │   ├── weeds.xml                 the 8 weeds, included by the scene
    │   └── assets/
    │       ├── ground/               grass + soil textures (Poly Haven, CC0)
    │       └── plants/{strawberry,sowthistle,shepherd}/
    │           ├── *_plant.png       texture
    │           ├── bands/            visual mesh, one per stalk segment
    │           └── collision/        CoACD convex pieces
    └── planning/
        ├── lerobot/                  submodule @ bad0260 (v0.6.1)
        └── weed_extraction/
            ├── weed_extraction/{__init__,env,launch}.py
            ├── configs/{view,capture}.yaml
            └── tests/{test_scene,test_env}.py
```

### Active scene: `workspace/application/field_description/scene_mobile_field.xml`

Includes `mobile_panda.xml` (robot) and `weeds.xml` (weeds). Owns the beds, the walking path, lighting, the 8 strawberries and the overview camera.

Compiled size: `nq=114 nv=106 nu=9 nbody=54 ngeom=375 nmesh=145 neq=9 ncam=3`.

Cameras (3): `rgb_camera` — worldbody, `mode="track" target="link0"`, third-person overview. `d435i_color` and `d435i_depth` — the RealSense D435i on the wrist, declared inside `mobile_panda.xml`.

## 5. What Works

| Component | Status |
|---|---|
| Scene loads and renders | Works, from any working directory, relative or absolute path. |
| Mobile base (M1) | Works. `base_slide_x` prismatic joint driven by the `base_x` position servo; reaches a commanded x within 0.01 m. |
| Wrist camera (M2) | Works. D435i model with colour and depth cameras at the datasheet FOVs, rigid to the `hand` body, looking down the gripper approach axis. |
| Plant physics (M3) | Works. Each weed is a free body held by a weld that `Roots` cuts at a sustained axial pull of 15 N. Only outward pull counts; sideways load bends the stalk, downward does nothing. |
| Flexible stalks | Works. Rigid bottom 4 cm, then 3 segments on sprung hinges at 3.0 Hz and damping ratio 0.15: droops 0.12 mm under gravity, bends 54 mm under 0.15 N, swings past upright twice before settling. |
| Textured plants (M4) | Works. All three species are textured meshes; no colour-based class cue remains. |
| Collision fidelity | Works. Visual meshes never collide; CoACD convex pieces do — 40 for the sowthistle, 20 for the shepherd's purse. |
| Gymnasium env | Works. `WeedExtraction-v0`, `info["uprooted"]` names weeds torn loose. No task defined. |
| Launcher | Works. `weed-launch view` / `capture`, or `run_scripts/simulate.sh`. |
| Interactive uprooting | Works. Ctrl+right-drag pulls a weed out at 28 mm of travel from any grab point. |
| Test suite | 14 tests pass. |

## 6. Limitations / Missing Components

- **No task.** Reward is 0 and `terminated` is always False. Deferred: the focus is a correct environment first.
- **No randomisation.** Every episode is the same field, so any policy trained on it would overfit.
- **No base motion controller.** The base is commanded by setting `ctrl["base_x"]` directly (M5).
- **Base y and orientation are fixed.** Translation along x only; the arm cannot reach both beds from one x-line.
- **No perception module.** Vision is available as observations, but there is no detector or deprojection (M6).
- **No IK, no scripted controller, no orchestration** (M7–M10). Not needed for RL; needed for demonstrations if VLA training follows.
- **Vision is slow**: 8.4 steps/s with one 128x128 camera against 97 state-only. Mesh decimation is the untried lever.
- **Leaves are rigid.** The stalk bends; individual leaves do not. The sowthistle mesh is 27 separable pieces and could support per-leaf hinges; the shepherd's purse is one welded surface and could not.
- **No soil model.** Beds are rigid boxes. Nothing can be dug, and root resistance is a constraint, not soil mechanics.
- **Sideways drags can still uproot** if pulled far enough, because the plant deflects until the cursor sits above it. 108 mm of travel versus 28 mm straight up.
- **No datasets, no policy code** despite `lerobot` being vendored.
- **Docker is unverified.** The image has never been built or run; the scripts are written but untested.
- **One asset is AI-generated.** The strawberry model came from Tripo; check its licence before the repo goes public.
- **The plant originals are gone.** Only the converted bands, collision pieces and textures are kept, so changing the band count or the decomposition means re-downloading the source models.

## 7. Milestones

- **M1 — Mobile base. DONE.**
- **M2 — Field scene + wrist camera. DONE.** Later replaced by the D435i model.
- **M3 — Plant physics: free joint + breakable root. DONE.**
- **M4 — Real textures, no colour cues. DONE.**
- **M5 — Base motion controller.** Drive to a target x and stop within tolerance.
- **M6 — Perception: weed versus crop from an image.** Bootstrap on ground-truth positions, then a real detector on the wrist camera; deproject to world.
- **M7 — Arm IK to a pre-grasp pose.** Open gripper ~5 cm above the detected point.
- **M8 — Grasp.** Descend, close, confirm hold through finger contact.
- **M9 — Pull and retreat.** Exceed the root threshold, verify the plant separates, return to a safe pose.
- **M10 — Orchestrator.** State machine over all weeds, never grasping a strawberry.

## 8. Technical Constraints and Decisions

**Simulator:** MuJoCo 3.8.1, native Python bindings. Pinned to `.venv`.

**Solver:** `timestep=0.001`, `cone="elliptic"`, `impratio=10`, Newton. Elliptic cones and a high impratio are what MuJoCo's documentation prescribes to stop a gripped object sliding free.

**No air.** MuJoCo's fluid model cost 17% of the step and changed neither fall time (248 ms over 30 cm, against a 247 ms ideal) nor tumble. Removed. Joint damping is not an alternative either: one value covers all six DOFs of a free joint, and anything large enough to slow a fall dwarfs a plant's rotational inertia, so the plant stops tumbling and sinks upright.

**M1 base:** slide joint `base_slide_x`, axis `1 0 0`, range `-2 4` m, damping 50, armature 1, frictionloss 1; base mass 30 kg; `<position>` actuator kp 2000, kv 300, forcerange ±1500 N.

**Actuator order:** `actuator1..7` (arm), `actuator8` (gripper, ctrlrange 0–255), `base_x`.

**D435i:** `pos="0.05 0 0.04"` on the `hand` body, `xyaxes="0 -1 0 1 0 0"`. Colour camera at x=+0.0325, depth at x=+0.0175, both 4.2 mm behind the glass; fovy 42° and 58° per the datasheet.

**Weed collision:** visual mesh is `contype=0 conaffinity=0 group=2`; collision is the CoACD decomposition at `group=3 condim=4 friction="1.5 0.02 0.001"`. MuJoCo replaces a mesh geom with its convex hull, which is 20–70 mm wide where the real stem is 1–2.5 mm — the fingers would close on empty air. Measured graspable width at the shepherd's stalk: 3.4 mm.

**Root anchorage:** one `<weld>` per weed, `anchor` at the planted position, `solref="0.01 1" solimp="0.95 0.99 0.001"`. `anchor` must be set explicitly; the default relpose drags the plant off its spot. Break force 15 N, sustained 20 ms, counted only along the planting axis. MuJoCo equalities have no break force, so `Roots.snap()` does it.

**Stalk springs:** hinge `stiffness=0.05`, `damping=8e-4`, `armature=1.4e-4`, `range=±0.5 rad`. The armature is what sets the sway — the segments' own inertia would put the stalk at 27 Hz, which reads as a twitch. The travel limit matters as much: at ±1.2 rad the plant folded flat and dropped its own tip 20 cm, which turned any sideways drag into an upward one.

**Plant masses:** sowthistle 15 g, shepherd's purse 8 g, set by explicit `<inertial>`. Mesh-derived inertia is not trustworthy — the meshes are not watertight.

**Interactive drag:** a critically damped spring sized to the grabbed body's mass, `ROOT_BREAK_FORCE / DRAG_TRAVEL` stiffness, clamped at 1.3×. MuJoCo's own perturbation scales with mass and gives a 15 g plant ~0.3 N. An undamped spring on a 2 g stalk tip rings at ~80 Hz and throws force spikes into the root.

**No artificial colour for classification** (hard requirement). `rgba` may be used for rendering realism only.

**Coordinate convention:** MuJoCo world frame, +z up, base travel along +x.

## 9. Software / Environment

**`.venv`** (Python 3.12, the only venv):
- `mujoco` 3.8.1, `gymnasium` 1.3.0, `numpy` 2.2.6, `opencv-python` 4.12.0.88, `draccus` 0.11.6 — all pinned in `pyproject.toml` to match the Docker image.
- `pytest` via the `[dev]` extra.
- `trimesh` and `coacd` were used to prepare the plant assets. They are not project dependencies; the generated meshes are committed.

**Docker (`vla_weed:latest`)** — Linux. Base `ros:jazzy-ros-base` (Ubuntu 24.04 supplies Python 3.12; the ROS base is for the ROS 2 packages this layout anticipates). Venv at `/opt/venv` with `--system-site-packages` so `rclpy` stays importable. The whole `workspace/` is copied in, so the image runs on its own; `devel.sh` mounts the repo over it for development and `base.sh` runs the image's own copy. The viewer draws on the host's X display: the run scripts call `xhost +local:root` and pass `DISPLAY` and the X socket. `MUJOCO_GL` is left unset so MuJoCo picks GLFW with a display and EGL without.

Scripts mirror the team's rviz2_web layout: `build.sh`, `base.sh`, `devel.sh`, `up.sh` (start and launch in one step), `enter_bash.sh`, `stop.sh`.

**Warnings:** the package installs a MuJoCo warning handler that prints to stderr, so no `MUJOCO_LOG.TXT` is ever written.

**Platform:** developed on macOS arm64 (Darwin 25.6). Native macOS OpenGL is 4.1, so `ARB_clip_control` is unavailable and MuJoCo prints a depth-accuracy warning on every offscreen depth render. Harmless, and absent on Linux.

## 10. Testing Requirements

Every milestone must record:

1. **Model compiles** — `nq`, `nu`, `nbody`, `ncam`, and any changed joint/actuator names.
2. **Physics sanity** — N steps with no NaN/Inf and no drift from `home` at rest.
3. **Kinematic reachability** — the target workspace is within arm reach from the base pose.
4. **A scripted behavioural check** for the milestone's feature.
5. **No regression** — `pytest` still passes.
6. **Headless-friendly** — verifiable without a GUI.

Run: `cd workspace/planning/weed_extraction && ../../../.venv/bin/python -m pytest`

---

## Maintenance Protocol (ONGOING RULE)

After **every** completed implementation step, update this file **before** starting the next. Append a dated entry with: milestone completed, files changed, implementation details, tests performed and results, current status, next milestone, and any decisions or parameters introduced. Also update the numbered sections above so the top of the document reflects reality.

