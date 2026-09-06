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
│   │   ├── mobile_panda.xml      # M1: Panda arm + gripper mounted on a mobile base (x-axis slide joint + `base_x` position servo)
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
├── docs/                         # empty
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

Compiled size: `nq=10 nv=10 nu=9 nbody=29 ncam=1 nmesh=70` (extra `base_slide_x`
DOF + `base_x` actuator vs. legacy; `nbody` is 3 lower than a naive copy because
the unused empty `*_template` bodies from the plant stubs are no longer included).
A non-colliding `base_cart` visual box is attached to `mobile_base`.

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
- **No wrist/hand camera** for close-range servoing; the only camera is a distant third-person view.
- ~~**Scene file location.**~~ Resolved in M1: the active scene is now `simulation/models/scene_mobile_field.xml` (project-owned). The vendored `scene.xml` is left untouched and still used by the legacy `src/` scripts.
- **Asset-path fragility (active scene: fixed; legacy scene: not).** MuJoCo prepends the inherited `meshdir` to relative mesh paths, which breaks the plant `<mesh file="../assets/plants/...">` entries in the stub XMLs. `scene_mobile_field.xml` works around this by declaring the plant meshes itself with **absolute** file paths (so it loads under any working dir). The **legacy** vendored `scene.xml` still has the bug and only compiles when loaded via an absolute path — the `src/` scripts do that via `Path(...).resolve()`. The absolute plant-mesh paths in `scene_mobile_field.xml` hardcode the repo location (`/home/divya/VLA-Weed-Extraction/...`); a cleaner fix (correcting `mobile_panda.xml`'s `meshdir`, or a shared `assetdir`) is deferred to M2.
- **No datasets / no policy code** despite `lerobot` being installed; `datasets/` is empty.

## 7. Planned Implementation Milestones (dependency order)

Each milestone must leave the project in a runnable, verifiable state. Do not start a milestone before its predecessor is complete and this file is updated.

- **M0 — Project scaffolding (no behaviour change).** _Folded into M1._
  A project-owned scene (`simulation/models/scene_mobile_field.xml`) now replaces in-place edits of the menagerie submodule. Venv pinned to `.venv` (mujoco 3.8.1).

- **M1 — Mobile base. ✅ DONE (2026-09-06, see Change Log).**
  `mobile_panda.xml` mounts the Panda on a `mobile_base` body with a prismatic `base_slide_x` joint (world x) and a `base_x` `<position>` servo; `scene_mobile_field.xml` is the new active scene. Arm + gripper actuators unchanged and preserved (`actuator1..8`).

- **M2 — Combined field scene + wrist camera.**
  One project-owned scene including the mobile Panda + both beds + plants. Add a `<camera>` on `hand` (wrist view); keep an overview camera. Verify both render.

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
