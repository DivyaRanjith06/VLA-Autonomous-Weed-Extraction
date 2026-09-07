# CLAUDE.md

Living technical record for this repository. Maintained by Claude Code. Update this file after every completed implementation step (see **Maintenance Protocol** at the end).

---

## 1. Project Title and Objective

**VLA-Weed-Extraction** — a MuJoCo simulation of an autonomous weeding robot for a raised-bed crop field.

Objective: build an end-to-end pipeline in which a mobile manipulator (Franka Emika Panda arm on a mobile base) drives along a crop bed, recognises weeds from **visual/textural** cues rather than artificial marker colours, and physically removes them from the soil while leaving the crop plants untouched.

## 2. Final End-to-End Goal

A single autonomous cycle, repeated for every weed in the bed:

1. Mobile base moves along the crop bed.
2. Robot identifies a target weed using visual / texture information (not artificial colour codes).
3. Base stops at the target weed.
4. Panda arm positions its gripper above the weed.
5. Gripper grasps the weed (stem).
6. Arm/base pulls the weed out of the soil (root resistance is overcome).
7. Robot retreats to a safe pose and advances to the next target; crop plants are never grasped.

## 3. Project Scope and Architecture

**In scope:** MuJoCo simulation only — robot model, field/scene model, plant physics, perception on rendered camera images, motion control (base + arm IK), grasp and extraction logic, and an orchestrating state machine. Optional: episode logging for a learned/VLA policy via the vendored `lerobot`.

**Out of scope (for now):** real hardware, sim-to-real transfer, training large VLA models.

**Intended runtime architecture (target, not yet built):**

```
                +------------------- orchestrator / state machine --------------------+
                |  select target -> drive base -> perceive -> IK above -> descend ->  |
                |  grasp -> pull -> retreat -> next                                   |
                +---------+----------------+----------------+-----------------+--------+
                          |                |                |                 |
                    base controller   perception       arm IK + grasp   plant/soil physics
                    (slide/planar     (segment weed     (Panda 7-DOF     (freejoint plant +
                     joint + actuator) vs crop from      + tendon         breakable root
                                       RGB/depth render) gripper)         constraint)
```

Simulation assets are split between:
- Project-authored models/assets under `simulation/`.
- The active scene file, which currently lives inside the vendored menagerie tree (`external/mujoco_menagerie/franka_emika_panda/scene.xml`) and has been locally modified.

## 4. Current Repository Structure

```
VLA-Weed-Extraction/
├── CLAUDE.md                     # this file
├── camera.png                    # sample render output committed to the repo
├── .gitignore
├── simulation/
│   ├── models/
│   │   ├── mobile_panda.xml      # M1: Panda arm + gripper mounted on a mobile base (x-axis slide joint + `base_x` position servo). M2: `wrist_camera` rigidly attached to the `hand` body (downward end-effector view).
│   │   ├── scene_mobile_field.xml # M1: project-owned field scene (beds + plants + lights + camera) that includes mobile_panda.xml — ACTIVE SCENE from M1 on. Declares the 3 plant meshes itself (absolute file paths) instead of `<include>`-ing the stub XMLs, so the plant assets load regardless of the Franka `meshdir` inherited from mobile_panda.xml.
│   │   ├── strawberry.xml        # stub: 1 mesh asset (STL, scale 0.18 -0.18 0.18) + empty template body. Unchanged; no longer included by the active scene (M1 path fix).
│   │   ├── sowthistle.xml        # stub: 1 mesh asset (STL, scale 0.05) + empty template body. Unchanged; not included by the active scene.
│   │   └── shepherd.xml          # stub: 1 mesh asset (STL, scale 0.40) + empty template body. Unchanged; not included by the active scene.
│   └── assets/
│       ├── ground/sparse_grass_diff_4k.png
│       ├── plants/
│       │   ├── strawberry/       # STL + source FBX + OBJ/MTL + textures (flower/leave/straw_leave .png)
│       │   ├── sowthistle/       # STL + source FBX + textures (Chardon_color/normal/ao/alpha .png)
│       │   ├── shepherd/         # STL + source GLTF + textures (2x .jpeg)
│       │   ├── weed.glb          # 0 bytes (empty placeholder)
│       │   └── strawberry.glb    # 0 bytes (empty placeholder)
│       └── mujoco_menagerie/     # git submodule (extra robot meshes; not used by current scene)
├── src/
│   ├── test_arm.py               # open-loop viewer demo: home pose -> hard-coded joint vector
│   ├── capture_image.py          # render one frame from camera "rgb_camera" -> camera.png
│   ├── live_camera.py            # passive viewer + OpenCV window streaming the camera feed
│   └── tools/camera_controller.py # empty file
├── external/
│   ├── mujoco_menagerie/         # git submodule
│   │   └── franka_emika_panda/
│   │       ├── panda.xml         # stock fixed-base Panda (included by scene.xml)
│   │       └── scene.xml         # ACTIVE SCENE, locally modified: "panda strawberry field"
│   └── lerobot/                  # git submodule, installed editable into .venv
├── scripts/                      # empty
├── datasets/                     # empty
├── docs/
│   └── m2/                        # M2 verification renders: wrist_camera_home.png, wrist_camera_over_bed.png
├── models/                       # empty
├── .venv/                        # Python venv (mujoco 3.8.1, lerobot, torch, ...) — gitignored
└── .venv_py310/                  # Python 3.10 venv (mujoco 3.11.0) — gitignored
```

Git: branch `main`, single commit `4f509ba "Initial project state"`.

### Active scene (from M1): `simulation/models/scene_mobile_field.xml`

Project-owned. A copy of the legacy scene below, changed **only** to (a) include
`mobile_panda.xml` (mobile base) instead of the vendored `panda.xml`, and (b) load
the plant meshes correctly. Beds, plant bodies, `rgba`s, lights, `visual`, skybox,
soil material and the `rgb_camera` are otherwise identical.

Asset-path handling: `mobile_panda.xml` sets a global `meshdir` (the Franka mesh
folder). MuJoCo prepends that `meshdir` to every *relative* mesh path — which
corrupts the plant paths in the stub XMLs (e.g.
`…/franka_emika_panda/assets/simulation/assets/plants/…`). Fix: the active scene
declares the three plant meshes in its own `<asset>` block with **absolute** file
paths (absolute paths bypass `meshdir`), and does **not** `<include>` the plant
stub XMLs. Result: the scene compiles whether loaded by absolute or relative path.
The Franka arm meshes are unaffected (bare filenames + the Franka `meshdir`).

Compiled size: `nq=10 nv=10 nu=9 nbody=29 ncam=2 nmesh=70` (extra `base_slide_x`
DOF + `base_x` actuator vs. legacy; `nbody` is 3 lower than a naive copy because
the unused empty `*_template` bodies from the plant stubs are no longer included;
`ncam=2` since M2 added `wrist_camera`). A non-colliding `base_cart` visual box is
attached to `mobile_base`.

Cameras (2): `rgb_camera` — worldbody, `mode="track" target="link0"`, distant
third-person overview (unchanged since the legacy scene). `wrist_camera` — M2,
declared inside the `hand` body in `mobile_panda.xml`, so it is rigid to the
end-effector; aimed along the gripper approach axis (hand +z), which points
straight down (world −z) in the `home` pose. `pos="0.12 0 0.2"` (hand frame):
0.20 m out along +z sits it ~7 cm past the fingertips so they never streak
across the frame, and the +12 cm along hand +x pushes the gripper mesh and its
cast shadow into one corner, leaving the bed surface / nearby plants filling
the rest of the view. `fovy="60"`. (Corrected 2026-09-07 — the first M2 value
`pos="0.05 0 0"` sat level with the fingers and they dominated the frame.)

### Legacy scene: `external/mujoco_menagerie/franka_emika_panda/scene.xml`

Vendored, fixed-base, unmodified. The three `src/` scripts still load this file
(resolved as `<repo>/external/mujoco_menagerie/franka_emika_panda/scene.xml`, absolute).
Contents:

- `<include>` of `panda.xml` (fixed-base Panda) and of the three plant stub XMLs (pulled in only for their `<mesh>` assets).
- Skybox (gradient), `soil` texture/material from `sparse_grass_diff_4k.png` (grass image), headlight + directional "sun" light.
- `path`: ground plane, `size="8 8 0.1"` at `z=-0.02`.
- `bed1`: box at `pos="0.80 -0.50 0.05"`, `size="2.5 0.35 0.05"` (10 m long in x, top surface z≈0.10).
- `bed2`: box at `pos="0.80 0.50 0.05"`, same size.
- 8 strawberries (`strawberry_1..8`), 4 sowthistle (`sowthistle_1..4`), 4 shepherd's-purse (`shepherd_1..4`), interleaved across both beds at `z=0.10`.
- Every plant is a **static `<body>` with a single `<geom type="mesh">` and no joint** → welded to the world.
- Plant classes are distinguished **only by `rgba`**, and the three greens are nearly identical:
  strawberry `0.15 0.55 0.12`, sowthistle `0.20 0.65 0.12`, shepherd `0.25 0.65 0.15`.
- One camera: `rgb_camera`, `mode="track"`, `target="link0"`, `pos="1.4 -0.8 0.9"` (third-person).

Verified compile (mujoco 3.8.1): `nq=9`, `nu=8`, `nbody=31`, `ncam=1`, **no free joints**.

## 5. Existing Components and What Currently Works

| Component | Status |
|---|---|
| Field scene loads & renders | Works. `scene.xml` compiles; renderer produces frames. |
| `src/capture_image.py` | Works. Renders one `rgb_camera` frame to `camera.png` (writes to CWD). |
| `src/live_camera.py` | Works. Passive viewer + OpenCV window of the camera feed. |
| `src/test_arm.py` | Works as a demo only. Opens viewer, sets `data.ctrl` to `home`, steps 1000, then to a hard-coded joint vector. No IK, no perception, no grasp logic. |
| Panda arm actuation | Works open-loop. 7 arm position actuators + 1 tendon-coupled gripper actuator (`actuator8`, `ctrlrange 0..255`). `home` keyframe present. |
| **Mobile base (M1)** | Works. `base_slide_x` prismatic joint (world x, range −2..4 m) driven by `base_x` position servo (`<position>` kp 2000, kv 300). Commanding `ctrl["base_x"]` translates the entire Panda along x; arm stays rigidly attached and holds its joint targets. Verified in `scene_mobile_field.xml`. |
| Two raised soil beds + plant layout | Present in the scene (static geometry). |
| **Wrist camera (M2)** | Works. `wrist_camera` fixed to the `hand` body, looking straight down along the gripper approach axis (hand +z ≈ world −z at `home`). Rigid to the end-effector (base/arm motion carries it exactly; verified drift ~1e-16 m in the hand frame). With the gripper posed over a bed, the offscreen render is a usable close-range top-down view: bed soil texture fills the frame, plants are clearly visible, the gripper mesh + shadow sit in one corner (~10 % of the frame). `rgb_camera` overview preserved and still renders. |
| Toolchain | Installed in `.venv`: `mujoco`, `lerobot`, `torch`, `gymnasium`, `gym-hil`, `opencv-python`, `pillow`, `numpy`. |

## 6. Current Limitations / Missing Components

- ~~**No mobile base.**~~ Resolved in M1: `mobile_panda.xml` now mounts the Panda on an x-axis slide joint with a `base_x` position servo; `scene_mobile_field.xml` is the project-owned scene that uses it.
- **No base motion controller.** M1 only adds the joint/actuator; there is no logic that chooses or drives to a target x yet (that is M5). The base is commanded directly by setting `ctrl["base_x"]`.
- **Base y and orientation are fixed.** M1 gives translation along x only (bed length). No cross-bed (y) motion or turning; the arm still cannot reach both beds from one x-line without an M7 IK reach study.
- **Plants cannot be picked or pulled.** No `freejoint` on any plant, no soil/root resistance model. Closing the gripper on a plant does nothing. (Goals 5–6 absent.)
- **Perception depends on artificial colour, and barely.** Classes differ only by three near-identical `rgba` greens. Plant meshes are STL (no UVs), so the real leaf/flower textures in `assets/plants/*/textures/` are never applied. (Goal 3 is currently the opposite of the requirement.)
- **No controller.** No inverse kinematics, no trajectory generation, no feedback control, no camera→world deprojection anywhere in the repo.
- **No orchestration.** `scripts/` is empty; there is no state machine.
- **Non-functional stubs.** `src/tools/camera_controller.py` is empty; `simulation/models/*.xml` are asset-only; `weed.glb` and `strawberry.glb` are 0 bytes.
- ~~**No wrist/hand camera.**~~ Resolved in M2: `wrist_camera` is fixed to the `hand` body for close-range viewing. Its aim is fixed geometry only (straight down the approach axis); no perception, deprojection, or servoing uses it yet (M6+).
- ~~**Scene file location.**~~ Resolved in M1: the active scene is now `simulation/models/scene_mobile_field.xml` (project-owned). The vendored `scene.xml` is left untouched and still used by the legacy `src/` scripts.
- **Asset-path fragility (active scene: fixed; legacy scene: not).** MuJoCo prepends the inherited `meshdir` to relative mesh paths, which breaks the plant `<mesh file="../assets/plants/...">` entries in the stub XMLs. `scene_mobile_field.xml` works around this by declaring the plant meshes itself with **absolute** file paths (so it loads under any working dir). The **legacy** vendored `scene.xml` still has the bug and only compiles when loaded via an absolute path — the `src/` scripts do that via `Path(...).resolve()`. The absolute plant-mesh paths in `scene_mobile_field.xml` hardcode the repo location (`/home/divya/VLA-Weed-Extraction/...`); a cleaner fix (correcting `mobile_panda.xml`'s `meshdir`, or a shared `assetdir`) is deferred to M2.
- **No datasets / no policy code** despite `lerobot` being installed; `datasets/` is empty.

## 7. Planned Implementation Milestones (dependency order)

Each milestone must leave the project in a runnable, verifiable state. Do not start a milestone before its predecessor is complete and this file is updated.

- **M0 — Project scaffolding (no behaviour change).** _Folded into M1._
  A project-owned scene (`simulation/models/scene_mobile_field.xml`) now replaces in-place edits of the menagerie submodule. Venv pinned to `.venv` (mujoco 3.8.1).

- **M1 — Mobile base. ✅ DONE (2026-09-06, see Change Log).**
  `mobile_panda.xml` mounts the Panda on a `mobile_base` body with a prismatic `base_slide_x` joint (world x) and a `base_x` `<position>` servo; `scene_mobile_field.xml` is the new active scene. Arm + gripper actuators unchanged and preserved (`actuator1..8`).

- **M2 — Combined field scene + wrist camera. ✅ DONE (2026-09-07, see Change Log).**
  `scene_mobile_field.xml` already combines the mobile Panda + both beds + plants. Added `<camera name="wrist_camera">` inside the `hand` body of `mobile_panda.xml`; the worldbody `rgb_camera` overview is preserved. Both cameras compile and render offscreen.

- **M3 — Plant physics: freejoint + breakable root.**
  Add `<freejoint>` to each plant and a breakable ground attachment (weld/connect equality with a tuned yield force, or an anchoring tendon/spring). Tune so a firm upward pull (order a few N) releases the plant. Verify by scripting a yank.

- **M4 — Real textures, remove colour cheats.**
  Replace STL plant geoms with textured meshes (OBJ/GLTF + `material`/`texture`); strawberry already has OBJ+MTL+textures. Remove class-discriminating `rgba`. Verify render shows visually distinct foliage with no colour coding.

- **M5 — Base motion controller.**
  Given a target x, drive the base actuator to it and stop within tolerance (open-loop first, closed-loop later). Satisfies goals 1–2 against ground-truth target positions.

- **M6 — Perception: weed vs crop from image.**
  Bootstrap with ground-truth body positions to unblock the pipeline, then add a real detector on the wrist-camera RGB (classical texture/shape features or a small learned model) that outputs a target pixel; deproject to world via known camera pose + bed-surface plane or a depth render.

- **M7 — Arm IK to pre-grasp pose.**
  Add an IK routine (Levenberg–Marquardt on the `hand` site, or `mink` / `dm_control`). Command joints so the open gripper sits ~5 cm above the detected point. Goal 4.

- **M8 — Grasp.**
  Descend along −z to stem height, close `actuator8`, confirm hold via finger contact/force. Goal 5.

- **M9 — Pull + retreat.**
  Raise base/arm past the M3 release threshold; verify the plant body separates and lifts; return to a safe pose. Goal 6 + step 7.

- **M10 — Orchestrator.**
  `scripts/run_weed_extraction.py`: state machine looping `select → drive → perceive → IK → descend → grasp → pull → retreat` over all weeds, never grasping strawberries. Optionally record episodes with `lerobot`.

## 8. Important Technical Constraints and Decisions

- **Simulator:** MuJoCo (native `mujoco` Python bindings + `mujoco.viewer`). **Pinned to `.venv` (mujoco 3.8.1)** as of M1; `.venv_py310` is not used.
- **M1 base parameters:** slide joint `base_slide_x`, axis `1 0 0`, range `-2 4` m, `damping=50`, `armature=1`, `frictionloss=1`; base `inertial mass=30`; actuator `base_x` = `<position>` `kp=2000 kv=300`, `ctrlrange="-2 4"`, `forcerange="-1500 1500"`; `ctrl["base_x"]` is the target base x in metres. `base_cart` geom is visual-only (`contype=0 conaffinity=0`).
- **Actuator order (M1 scene):** `actuator1..7` (arm), `actuator8` (gripper), `base_x` (index 8). Arm/gripper indices are unchanged from the legacy model; the base actuator was appended.
- **M2 wrist camera parameters:** `<camera name="wrist_camera">` in the `hand` body of `mobile_panda.xml`; `pos="0.12 0 0.2"` (hand frame), `xyaxes="1 0 0 0 -1 0"` (camera x = hand +x, camera −z = hand +z, so it looks along the gripper approach axis), `fovy="60"`, default `mode="fixed"`. In the `home` pose the optical axis is world −z (straight down); verified `view_dir.z = −1.000`. The 0.20 m +z component clears the fingertips (~0.108 m in the hand frame) so they never streak the frame; the 0.12 m +x component offsets the gripper mesh + cast shadow into a corner. No actuator, joint, DOF, or `home` keyframe change — camera-only. **First value `pos="0.05 0 0"` was rejected** (sat at finger-base height → gripper filled the view); corrected 2026-09-07.
- **qpos order (M1 scene):** `[base_slide_x, joint1..7, finger_joint1, finger_joint2]` (`nq=10`). `home` keyframe updated accordingly.
- **Robot:** Franka Emika Panda, 7-DOF arm + parallel gripper. Gripper is tendon-coupled via `actuator8` with a remapped `ctrlrange="0 255"` (0 = closed target, 255 = open target). Arm actuators are position-servo `general` actuators.
- **Angles:** radians (`compiler angle="radian"`).
- **Reach:** fixed-base Panda horizontal reach ≈ 0.85 m; this is the reason M1 (mobile base) precedes any grasping work.
- **Bed geometry (current scene):** beds at `y = ±0.50`, half-thickness in y = 0.35, top surface `z ≈ 0.10`, extent in x ≈ `[-1.70, 3.30]`. Plants rest at `z = 0.10`.
- **No artificial colour for classification** (hard requirement): perception must work from geometry/texture. `rgba` may still be used for non-cued rendering realism only.
- **Do not rely on editing the menagerie submodule.** From M0, the authoritative scene is a project-owned file; the submodule stays vendored/unmodified where practical.
- **Plant extraction model:** plants get a `freejoint` plus a breakable constraint representing root anchorage; the break force is a tunable parameter recorded in this file when introduced.
- **Coordinate convention:** MuJoCo world frame, +z up, base travel along +x (bed length).

## 9. Current Software / Environment Information

**`.venv`** (primary; used by `src/` scripts):
- `mujoco` 3.8.1
- `lerobot` 0.6.1 (editable install from `external/lerobot`)
- `torch` 2.11.0, `torchvision` 0.26.0, `torchcodec` 0.11.1
- `numpy` 2.2.6, `opencv-python` 4.12.0.88, `pillow` 12.3.0
- `gymnasium` 1.3.0, `gym-hil` 0.1.14, `aruco-opencv-msgs` 2.3.1

**`.venv_py310`** (Python 3.10): `mujoco` 3.11.0 (minimal). Not used — project is pinned to `.venv` since M1.

**Platform:** Linux 6.8 x86_64, bash. Git submodules: `external/mujoco_menagerie`, `external/lerobot`, `simulation/assets/mujoco_menagerie`.

**Rendering note:** `capture_image.py` writes `camera.png` into the current working directory; `live_camera.py` opens a GUI window (needs a display).

## 10. Testing / Verification Requirements

Every milestone must include, and this file must record, the following:

1. **Model compiles:** `mujoco.MjModel.from_xml_path(<scene>)` succeeds; record `nq`, `nu`, `nbody`, `ncam`, and joint/actuator names that changed.
2. **Physics sanity:** `mujoco.mj_step` runs N steps with no `NaN`/`Inf` in `data.qpos`/`data.qvel` and no unexpected divergence from the `home` keyframe at rest.
3. **Kinematic reachability (M1+):** verify the target workspace (bed surface at the relevant x) is within arm reach from the base pose.
4. **Behavioural check for the milestone's feature**, scripted and repeatable:
   - M1/M5: base reaches commanded x within tolerance (state the tolerance).
   - M3/M9: plant releases under the target pull force and not under smaller disturbances.
   - M4: rendered image inspected; confirm no colour-based class cue remains.
   - M6: detector precision/recall (or at least confusion counts) on a fixed set of rendered frames.
   - M7: end-effector position error vs. target < stated tolerance.
   - M8: grasp holds through a lift of stated height.
   - M10: full loop clears all weeds, grasps zero strawberries, over a fixed scene seed.
5. **No regression:** the three existing `src/` scripts (or their successors) still run.
6. **Headless-friendly:** provide a way to verify without a GUI (offscreen render or state assertions), since viewer windows need a display.

Record exact commands run and their output/results in the Maintenance Protocol log.

---

## Maintenance Protocol (ONGOING RULE)

After **every** completed implementation step, update this file **before** starting the next step. Append a dated entry to the log below containing:

- **Milestone completed**
- **Files changed** (created / modified / deleted)
- **Implementation details** (what was built and how)
- **Tests performed and results** (exact commands + outcomes)
- **Current project status**
- **Next milestone**
- **Decisions / parameters introduced** (e.g. break force, tolerances, pinned versions)

Also update the relevant numbered sections above (structure, what works, limitations, environment) so the top of the document always reflects reality.

### Change Log

#### 2026-09-06 — Document created (pre-implementation)

- **Milestone completed:** none. Initial inspection and creation of this record.
- **Files changed:** created `CLAUDE.md`. No other files touched.
- **Implementation details:** none. Repository inspected read-only: `simulation/`, `src/`, plant XMLs, the active scene `external/mujoco_menagerie/franka_emika_panda/scene.xml`, `panda.xml`, both venvs. Ran one read-only compile check of the scene with `.venv` mujoco 3.8.1 → `COMPILES OK`, `nq=9 nu=8 nbody=31 ncam=1`, no free joints.
- **Tests performed and results:** compile check only (above). No behavioural tests.
- **Current project status:** scene + arm render and actuate open-loop; no mobile base, no plant physics, no perception, no IK, no orchestration. See sections 5–6.
- **Next milestone:** **M0 — project scaffolding** (project-owned scene path + entry-point layout; pin a MuJoCo/venv version).
- **Decisions / parameters introduced:**
  - This file is the single living technical record; update rules above.
  - Milestone plan M0–M10 fixed in section 7.
  - Constraint reaffirmed: no artificial-colour classification (section 8).
  - Open decision for M0: pin `.venv` (mujoco 3.8.1) vs `.venv_py310` (mujoco 3.11.0).

#### 2026-09-06 — M1: mobile base (x-axis)

- **Milestone completed:** M1 (M0 scaffolding folded in).
- **Files changed:**
  - **Modified** `simulation/models/mobile_panda.xml` (was an unused fixed-base copy of the stock Panda). 6 surgical edits: (1) model name `panda` → `mobile_panda`; (2) `meshdir` → `../../external/mujoco_menagerie/franka_emika_panda/assets` so the arm meshes resolve without a local copy; (3) wrapped `link0` in a new `<body name="mobile_base">` carrying `<joint name="base_slide_x" type="slide" axis="1 0 0" range="-2 4" damping="50" armature="1" frictionloss="1"/>`, an `<inertial mass="30">`, and a visual-only `base_cart` box geom; (4) matching `</body>` added before `</worldbody>`; (5) appended `<position name="base_x" joint="base_slide_x" kp="2000" kv="300" ctrlrange="-2 4" forcerange="-1500 1500"/>` to `<actuator>`; (6) `home` keyframe extended to `nq=10`/`nu=9` (`base_slide_x` prepended to qpos, `base_x`=0 appended to ctrl). `link0` subtree left un-reindented to keep the diff minimal.
  - **Created** `simulation/models/scene_mobile_field.xml` — copy of the vendored `franka_emika_panda/scene.xml` with only: `include panda.xml` → `include mobile_panda.xml`; plant includes and the soil-texture path rewritten relative to `simulation/models/`. Beds, plants, `rgba`s, lights, `visual`, skybox, and the `rgb_camera` (still `mode="track" target="link0"`) are byte-identical.
  - **Not touched:** `external/mujoco_menagerie/**` (vendored `scene.xml` still compiles and is still what the three `src/` scripts load), all `src/` files, plant stub XMLs.
- **Implementation details:** the whole Panda is now a child of `mobile_base`, so `base_slide_x` translates arm + gripper + wrist rigidly along world x. `base_x` is a PD position servo; `data.ctrl[<base_x>]` is the target base x in metres. First attempt used a bare `<general>` actuator with no `biastype` → it defaulted to `biastype="none"` (open-loop force) and slammed the joint to its range limit; switched to `<position>` which fixed it.
- **Tests performed and results** (`.venv` mujoco 3.8.1, headless, script in scratchpad):
  - Compile: `scene_mobile_field.xml` loads via absolute path. `nq=10 nv=10 nu=9 nbody=32 ncam=1`. Joints: `base_slide_x, joint1..7, finger_joint1, finger_joint2`. Actuators: `actuator1..8, base_x` (arm/gripper indices unchanged).
  - Preserved: `rgb_camera` present; geoms `path/bed1/bed2` present; 16 plant bodies present.
  - Motion: from `home`, command `ctrl["base_x"]=2.0`, hold arm, step 6 s → `base_slide_x` settles at **2.0002 m**; `link0` and `hand` both translate **+2.00 m** in x; arm joints hold home within **0.007 rad**; `|hand−link0|` 0.835→0.831 m (rigid). Reverse command `−1.5` → settles **−1.5001 m**; all `qpos/qvel` finite. **ALL PASS.**
  - Regression: vendored `scene.xml` still compiles via absolute path (as `src/` scripts load it); no `src/`/`scripts/` file references the new files.
- **Current project status:** mobile base works (x translation, direct `ctrl`). Still no controller/perception/IK/grasp/pull/weed-physics/textures. Active scene for future work: `simulation/models/scene_mobile_field.xml`.
- **Next milestone:** **M2 — combined field scene + wrist camera** (add a `<camera>` on `hand`; keep an overview camera; verify both render). Note: much of M2's "combined scene" already exists as `scene_mobile_field.xml`, so M2 is mostly the wrist camera + a render check.
- **Decisions / parameters introduced:**
  - Venv pinned to `.venv` (mujoco 3.8.1); `.venv_py310` unused.
  - Base motion is **x-only** for now (bed length); no y / yaw.
  - Base actuator uses MuJoCo `<position>` (kp 2000, kv 300); joint range `-2..4 m`; `forcerange ±1500 N`; base mass 30 kg, `armature 1`.
  - `base_cart` geom is cosmetic and collision-free to avoid disturbing bed/ground contact.
  - Minimal-diff convention: wrap, don't reindent, large subtrees.

#### 2026-09-06 — M1 fix + verification: plant asset-path resolution

- **Milestone:** M1 follow-up. Verification had failed: `scene_mobile_field.xml` loaded via a *relative* path could not open the plant STLs — MuJoCo prepended the Franka `meshdir` (from `mobile_panda.xml`) to the stub XMLs' relative mesh paths, producing `…/franka_emika_panda/assets/simulation/assets/plants/strawberry/strawberry.stl`. (The vendored legacy `scene.xml` has the identical bug; it only works because the `src/` scripts pass absolute paths.)
- **Files changed:**
  - **Modified** `simulation/models/scene_mobile_field.xml` only: removed the three `<include file="{strawberry,sowthistle,shepherd}.xml"/>` lines; added an `<asset>` block declaring `strawberry_mesh` / `sowthistle_mesh` / `shepherd_mesh` with the **same scales** as the stubs (`0.18 -0.18 0.18` / `0.05 0.05 0.05` / `0.40 0.40 0.40`) and **absolute** `file` paths under `/home/divya/VLA-Weed-Extraction/simulation/assets/plants/…`. Absolute paths bypass `meshdir`. Updated the file's header comment.
  - **Not changed:** `mobile_panda.xml` (robot / base / actuators untouched), the plant stub XMLs (unchanged on disk; the active scene simply no longer includes them — their only content was one `<mesh>` each, now declared in the scene, plus an unused empty `*_template` body), `external/**`, `src/**`, `CLAUDE.md` milestone plan.
- **Tests performed (`.venv`, MuJoCo 3.8.1):**
  - Compile — **relative** path from repo root: OK. `nq=10 nv=10 nu=9 nbody=29 ncam=1 nmesh=70`.
  - Compile — **absolute** path with cwd `/tmp`: OK, identical stats.
  - Compile — relative path from `src/` cwd: OK.
  - Full M1 behaviour re-run (absolute): joints `base_slide_x, joint1..7, finger_joint1/2`; actuators `actuator1..8, base_x` (arm/gripper order preserved); `rgb_camera` + `path/bed1/bed2` + 16 plant bodies present. From `home`, `ctrl["base_x"]=2.0` → `base_slide_x` = **2.0002 m**, `link0` & `hand` translate **+2.00 m** x, arm joints hold within **0.007 rad**, arm rigid wrt base (`|hand−link0|` 0.835→0.831). Reverse to **−1.5** → **−1.5001 m**, state finite. **ALL PASS.**
  - Regression: vendored `scene.xml` still compiles via absolute path.
- **Current project status:** M1 complete and verified. `scene_mobile_field.xml` compiles under any working directory; mobile base translates in x with the Panda attached; arm + gripper actuators unchanged. No perception / textures / IK / grasp / pull / weed physics.
- **Next milestone:** **M2 — combined field scene + wrist camera.** Also revisit the hardcoded absolute plant-mesh paths (fix `mobile_panda.xml`'s `meshdir` or introduce a shared `assetdir`) so the scene is location-independent.
- **Decisions / parameters introduced:**
  - Plant meshes are declared in `scene_mobile_field.xml` with absolute paths; plant stub XMLs are retained on disk but unused by the active scene.
  - `nbody` for the active scene is 29 (the 3 empty `*_template` bodies are gone); 16 plant bodies + robot chain + world/beds intact.
  - Known debt (M2): absolute paths hardcode the repo location.

#### 2026-09-07 — M2: wrist / end-effector camera

- **Milestone completed:** M2 (wrist camera; the "combined field scene" half was already satisfied by `scene_mobile_field.xml` from M1).
- **Files changed:**
  - **Modified** `simulation/models/mobile_panda.xml` — one addition: a `<camera name="wrist_camera" pos="0.05 0 0" xyaxes="1 0 0 0 -1 0" fovy="60"/>` element inside `<body name="hand">`, placed right after the `hand_c` collision geom and before `<body name="left_finger">`, with an explanatory comment. Nothing else touched — no change to bodies, joints, DOFs, actuators, tendons, equalities, or the `home` keyframe.
  - **Not touched:** `simulation/models/scene_mobile_field.xml` (its worldbody `rgb_camera` is unchanged and still the overview), all plant/scene geometry, `external/**` (vendored `scene.xml` unchanged), `src/**`, `scripts/**`.
- **Implementation details:** the camera is declared as a child of the `hand` body, so it is rigidly welded to the end-effector and inherits both base translation (M1) and future arm motion. Orientation `xyaxes="1 0 0 0 -1 0"` sets the camera frame so its viewing axis (−z of the camera frame) coincides with the hand's +z axis — the gripper approach / grasp direction. In the Panda `home` ready pose the hand +z points straight down, so the wrist camera looks vertically onto the bed surface just beyond the fingertips. The +x (hand-frame) 5 cm offset shifts the fingers to the edge of the image so most of the frame is bed/plants. `fovy=60` widens the close-range view. `mode` left at the default `fixed` (pose fully determined by the parent body; no tracking target needed). This is passive sensor geometry only — no perception, IK, servoing, grasping, pulling, or autonomous motion was added.
- **Tests performed and results** (`.venv`, MuJoCo 3.8.1, headless, `MUJOCO_GL=egl`; script `scratchpad/m2_test.py`):
  - **Compile:** `scene_mobile_field.xml` loads. `nq=10 nv=10 nu=9 nbody=29 ncam=2 nmesh=70` — only `ncam` changed vs. M1 (1 → 2). Cameras: `['rgb_camera', 'wrist_camera']`. `wrist_camera` `cam_bodyid` → body `hand`, `mode=0` (fixed).
  - **M1 preserved:** joints `['base_slide_x','joint1'..'joint7','finger_joint1','finger_joint2']`; actuators `['actuator1'..'actuator8','base_x']` — unchanged order/count. Physics: 2000 `mj_step` from `home`, all `qpos`/`qvel` finite, `max|qvel| ≈ 6e-6` (at rest).
  - **Rigid attachment:** `cam_xpos − hand_xpos` offset = `[~0, 0.05, ~0]` at base x=0; after commanding `base_x → 2.0` and stepping 4000×, offset drift = `1.7e-5` m; hand moved `dx=2.002`, camera moved `dx=2.002` (moves with the end-effector).
  - **Aim:** at `home`, `wrist_camera` world view direction = `[0, 0, -1.000]` → straight down toward the bed. Assertion `view_dir.z < -0.5` passes.
  - **Render:** offscreen `mujoco.Renderer` 640×480. `wrist_camera` → `640×480×3 uint8`, mean 104, 100 % non-black; image shows the bed soil texture from above with the gripper fingers at the left edge and a plant in-frame (`scratchpad/m2_wrist_camera.png`). `rgb_camera` also renders unchanged (`scratchpad/m2_rgb_camera.png`, mean 101).
  - **Regression:** vendored `external/.../franka_emika_panda/scene.xml` still compiles (`nq=9 nu=8 nbody=31 ncam=1`); no `src/` script references `mobile_panda.xml`.
  - Result: **ALL M2 CHECKS PASSED.**
- **Current project status:** mobile base (M1) + wrist camera (M2) both working and verified. The active scene `scene_mobile_field.xml` now exposes two cameras: `rgb_camera` (overview) and `wrist_camera` (end-effector, downward). Still no plant physics, textures, perception, IK, grasp, pull, base motion controller, or orchestration.
- **Next milestone:** **M3 — plant physics: `<freejoint>` + breakable root** (add a `freejoint` to each plant body and a breakable ground attachment tuned so a few-newton upward pull releases the plant; verify with a scripted yank).
- **Decisions / parameters introduced:**
  - `wrist_camera` lives in `mobile_panda.xml` (the robot model), not the scene file, so any scene that includes the mobile Panda gets the end-effector view for free.
  - Camera aim is fixed geometry along the gripper approach axis (hand +z); no dynamic tracking. Params: `pos="0.05 0 0"`, `xyaxes="1 0 0 0 -1 0"`, `fovy="60"`.
  - `rgb_camera` is the designated overview camera and is left exactly as inherited from the legacy scene.

#### 2026-09-07 — M2 correction: wrist camera placement

- **Milestone:** M2 follow-up (no new milestone started). The first M2 wrist-camera render was judged not useful — `pos="0.05 0 0"` put the camera at finger-base height looking parallel to the fingers, so the gripper/fingers and their shadows streaked across and dominated the frame.
- **Files changed:**
  - **Modified** `simulation/models/mobile_panda.xml` — the `wrist_camera` element only: `pos` `"0.05 0 0"` → `"0.12 0 0.2"` (hand frame); `xyaxes="1 0 0 0 -1 0"` and `fovy="60"` unchanged; comment updated. No other line touched. Still a child of `<body name="hand">` (rigid to the end-effector); orientation still straight down the gripper approach axis.
  - **Not touched:** `scene_mobile_field.xml` (`rgb_camera` overview unchanged), `external/**`, `src/**`, all M1 robot/base/actuator/keyframe content.
- **Why this placement:** raycast + render diagnostics showed (a) the fingertips reach ≈0.108 m along hand +z, so a camera at `z=0` looks straight along them → edge-on streaks; (b) the wide dark bands in the first render were the arm/gripper **shadows** cast onto the ground (confirmed: they vanish with `mjRND_SHADOW` off); (c) in the `home` park pose the TCP hangs over the ~0.3 m central walkway between the two beds, not over a bed. Fix: move the camera 0.20 m out along +z (≈7 cm past the fingertips — no more finger streaks) and 0.12 m along hand +x (shifts the gripper mesh + its shadow into one image corner). Orientation left straight-down: correct for eye-in-hand servoing, where the optical axis should track the approach vector rather than being permanently canted at one bed.
- **Tests performed and results** (`.venv`, MuJoCo 3.8.1, headless `MUJOCO_GL=egl`; `scratchpad/m2_test.py`):
  - **Compile:** `scene_mobile_field.xml` loads. `nq=10 nv=10 nu=9 nbody=29 ncam=2 nmesh=70` (unchanged from the first M2). Cameras `['rgb_camera','wrist_camera']`; `wrist_camera` → body `hand`, `mode=fixed`.
  - **M1 preserved:** joints and actuators identical list/order; 2000 `mj_step` from `home` all finite, `max|qvel|≈6e-6`.
  - **Rigid attach:** camera position in the hand body frame = `[0.12, 0, 0.20]` exactly; after `base_x → 2.0` + 4000 steps the hand-frame offset drift is `1.6e-16` m (world-frame offset moves ~1e-3 m, but that is just arm-joint flex under the base accel, not the camera).
  - **Aim:** `home`-pose world view direction `[0, 0, -1.000]` (straight down).
  - **Render — both cameras, `home` pose:** `rgb_camera` mean 101.5, unchanged. `wrist_camera` mean 103.1, 100 % non-black (`docs/m2/wrist_camera_home.png`). At `home` the wrist view is over the bare walkway with arm shadows down the left ~third — a park-pose artifact, not representative of operation.
  - **Render — `wrist_camera`, gripper posed over bed2** (static `qpos`: `joint1 = 0.85` rad; **no controller/IK added**, this is only to place the TCP over a bed for the usefulness check): `docs/m2/wrist_camera_over_bed.png`. Bed soil texture fills the frame, a weed is clearly visible (bottom-left), the gripper mesh is a small wedge in the top-right corner with a modest shadow. Heuristic: scene-content (ground+plant mid-tones) = **99.6 %** of the frame; bright-green (plants + gripper edge) = 8.5 %. Assertions `green ∈ (2,45) %` and `content > 60 %` pass.
  - **Regression:** vendored `external/.../scene.xml` still compiles (`nq=9 nu=8 nbody=31 ncam=1`); no `src/` script references `mobile_panda.xml`.
  - Result: **ALL M2 CHECKS PASSED.** The over-bed render is a usable input for future weed perception (M6).
- **Current project status:** M1 (mobile base) + M2 (wrist camera, corrected) working and verified. Active scene `scene_mobile_field.xml` exposes `rgb_camera` (overview) + `wrist_camera` (end-effector, straight-down, gripper in-corner). No plant physics, textures, perception, IK, grasp, pull, base controller, or orchestration.
- **Next milestone:** **M3 — plant physics: `<freejoint>` + breakable root** (not started).
- **Decisions / parameters introduced:**
  - Final `wrist_camera` params: `pos="0.12 0 0.2"`, `xyaxes="1 0 0 0 -1 0"`, `fovy="60"`, in the `hand` body.
  - Wrist camera stays straight-down (approach-axis aligned), not canted toward a bed — the eye-in-hand convention for later servoing.
  - The `home` park pose leaves the TCP over the central walkway; a representative wrist view requires the arm/base to first position the gripper over a bed (that positioning logic is M5/M7, not M2).
