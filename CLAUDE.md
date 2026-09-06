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
│   │   ├── mobile_panda.xml      # byte-for-byte copy of stock Franka panda.xml; MISNAMED (no base); unused
│   │   ├── strawberry.xml        # stub: 1 mesh asset (STL, scale 0.18 -0.18 0.18) + empty template body
│   │   ├── sowthistle.xml        # stub: 1 mesh asset (STL, scale 0.05) + empty template body
│   │   └── shepherd.xml          # stub: 1 mesh asset (STL, scale 0.40) + empty template body
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

### Active scene: `external/mujoco_menagerie/franka_emika_panda/scene.xml`

All three `src/` scripts load this exact file (resolved as `<repo>/external/mujoco_menagerie/franka_emika_panda/scene.xml`). Contents:

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
| Two raised soil beds + plant layout | Present in the scene (static geometry). |
| Toolchain | Installed in `.venv`: `mujoco`, `lerobot`, `torch`, `gymnasium`, `gym-hil`, `opencv-python`, `pillow`, `numpy`. |

## 6. Current Limitations / Missing Components

- **No mobile base.** `mobile_panda.xml` is an identical copy of the stock fixed-base `panda.xml` and is referenced by nothing. `link0` is welded to the world at the origin. Nothing translates along the bed. (Goal 1 absent.)
- **Arm cannot reach most of the field.** Beds are ~10 m long in x; a fixed base at `x=0` with ~0.85 m Panda reach covers only the few plants near the origin. (Goal 4 blocked without base motion.)
- **Plants cannot be picked or pulled.** No `freejoint` on any plant, no soil/root resistance model. Closing the gripper on a plant does nothing. (Goals 5–6 absent.)
- **Perception depends on artificial colour, and barely.** Classes differ only by three near-identical `rgba` greens. Plant meshes are STL (no UVs), so the real leaf/flower textures in `assets/plants/*/textures/` are never applied. (Goal 3 is currently the opposite of the requirement.)
- **No controller.** No inverse kinematics, no trajectory generation, no feedback control, no camera→world deprojection anywhere in the repo.
- **No orchestration.** `scripts/` is empty; there is no state machine.
- **Non-functional stubs.** `src/tools/camera_controller.py` is empty; `simulation/models/*.xml` are asset-only; `weed.glb` and `strawberry.glb` are 0 bytes.
- **No wrist/hand camera** for close-range servoing; the only camera is a distant third-person view.
- **Scene file location.** The active scene lives inside the vendored `external/mujoco_menagerie` submodule and has been edited in place — fragile and outside project-owned paths.
- **Latent asset-path fragility.** Plant `<mesh file="../assets/plants/...">` paths resolve against `meshdir="assets"` inherited from `panda.xml`; correct only from the current include location.
- **No datasets / no policy code** despite `lerobot` being installed; `datasets/` is empty.

## 7. Planned Implementation Milestones (dependency order)

Each milestone must leave the project in a runnable, verifiable state. Do not start a milestone before its predecessor is complete and this file is updated.

- **M0 — Project scaffolding (no behaviour change).**
  Establish a project-owned scene path and an entry-point script layout so later work does not keep editing the menagerie submodule. Decide whether to relocate/mirror `scene.xml` under `simulation/models/`. No robot/behaviour changes.

- **M1 — Mobile base.**
  Author a proper mobile Panda model: a base body carrying a prismatic (`slide`) joint along the bed axis (x) — optionally a second slide (y) or a planar joint — with position actuators, Panda mounted on top at a working height. Confirm the arm can reach the bed surface from the base.

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

- **Simulator:** MuJoCo (native `mujoco` Python bindings + `mujoco.viewer`). Two venvs exist (3.8.1 and 3.11.0); pin one before M1 — see Maintenance Protocol log.
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

**`.venv_py310`** (Python 3.10): `mujoco` 3.11.0 (minimal).

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
