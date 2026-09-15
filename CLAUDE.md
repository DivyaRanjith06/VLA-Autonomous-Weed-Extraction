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
├── run_scripts/
│   ├── simulate.sh           launcher; works natively and inside the container
│   └── test_up_root.sh       scripted grasp-twist-pull demo in the viewer
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
            ├── weed_extraction/{__init__,env,ik,launch,uproot}.py
            ├── configs/{view,capture,uproot}.yaml
            └── tests/{test_scene,test_env,test_ik,test_uproot}.py
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
| Collision-aware IK (M7) | Works. mink differential IK on a plant-free copy of the scene; 7 iterations and 0.003 mm to a reachable pose, ~0.25 ms per solve. Keeps the arm off the beds, the ground and its own shoulder. |
| Root torsion | Works. Pull and twist share one linear failure criterion, so twisting a weed loose takes less pull than hauling it out. |
| Scripted extraction (M8-M9) | Works for the shepherd's purse. Side grasp at 40 deg off the stalk's axis, 16 mm up; turn 10 deg about the world vertical, then lift. shepherd_1 comes out during the lift at 11.8 s on about 52% pull and 66% twist -- neither alone is enough. The plant never enters the gripper by more than 0.33 mm and drops the instant the jaws open. |
| Sowthistle approach | Works. The hand and wrist camera pass through every sowthistle segment, so all four sowthistles reach the grip still rooted. |
| Grasp registration | Works. A grasp counts when both fingertip pads touch the plant, or when one pad is on the plant's stem capsule and the other is within 1.0 mm of it. All four sowthistles register as held and are carried out of the bed. |
| Sowthistle extraction | Partial. The root breaks during the twist, before any meaningful pull -- see "Current known limitation" in §6. |
| Test suite | 36 tests pass. |

## 6. Limitations / Missing Components

### Current known limitation: sowthistle roots break on the twist alone

- **What happens.** All four sowthistles are grasped, but their roots break during the twist phase at about 0.4–0.5 N of pull and 0.070-0.074 N.m of twist. The limit is 15 N of pull or 0.07 N.m of twist, combined linearly.
- **What that means.** The twist does all the work. The pull that the combined criterion is meant to share the load with never develops.
- **What the lift does.** It carries a plant whose root has already broken. The plant leaves the bed, but this is not a combined pull-and-twist extraction.
- **What is not affected.** The shepherd's purse still comes out during the lift on pull and twist together.
- **Not yet investigated.** Why the sowthistle's root sees the full twist torque after only a few degrees of turn.

### Other limitations

- **No task.** Reward is 0 and `terminated` is always False. Deferred: the focus is a correct environment first.
- **No randomisation.** Every episode is the same field, so any policy trained on it would overfit.
- **No base motion controller.** The base is commanded by setting `ctrl["base_x"]` directly (M5).
- **Base y and orientation are fixed.** Translation along x only; the arm cannot reach both beds from one x-line.
- **No perception module.** Vision is available as observations, but there is no detector or deprojection (M6).
- **No orchestration** (M10). One weed at a time; nothing sequences the field.
- **The sowthistle's stem is a stand-in.** Its CoACD pieces near the base are leaf blobs reaching 28 mm from the axis, so the fingers grip a manual capsule on the stalk instead, as they do on the shepherd's purse. It is a hand-placed shape on the body's axis, not a decomposition of the real stem.
- **The grasp height has little margin.** The arm settles about 2 mm below its commanded tool point. At the old 14 mm the fingertip pads touched the bed at every weed except shepherd_1. At `GRASP_HEIGHT` = 16 mm the pads clear the bed by 1.7-2.5 mm across all eight weeds. That was measured once, not by a test.
- **The grasp is a weld, not friction.** See §8. The condition is geometric, but the hold itself is modelled rather than simulated.
- **The fingertips pass through leaves.** They only collide with the stalk. Real leaves are thin and would be brushed aside; ours are rigid convex lumps, and one 22 mm out would otherwise jam the jaws before they reached the stem.
- **The hand and wrist camera pass through the sowthistles.** For the same reason: the sowthistle canopy is wide enough that the hand otherwise shoves the plant over during the approach. The shepherd's purse keeps full collision with both.
- **The fingertips also pass through the strawberries**, which carry no grippable bit. The crop cannot be grasped at all, rather than being grasped and rejected. Harmless until M10 needs the robot to demonstrate leaving it alone.
- **The weed hangs on the pads for 0.58 s after the fingers open.** That is the price of the squash: the pads are 8.5 mm inside the stem and have to clear it before the plant can fall. Less compliance releases faster but cannot grip -- see the note on solimp in §8.
- **40 deg off the stalk's axis is as side-on as the arm gets.** Two separate limits, and they are easy to confuse. The gripper *fits* at any angle up to 75 deg -- measured, the deepest part is only 12-16 mm below the tool point until 90 deg, where link6 swings down 49 mm. But *getting there* is the binding constraint: past 40 deg there is no path down to the stalk that keeps the fingers off the bed, and the solve stalls 70-110 mm short. Measured distance from the tool point to the stalk at the end of the approach: 1.1 mm at 40 deg, 68 mm at 60 deg, 113 mm at 75 deg, at every grasp height from 14 to 40 mm. A stalled approach is not an obvious failure -- the gripper closes on nothing and then sinks toward the target during the turn, which looks like a grasp that slips. A true level grasp needs a purpose-built end effector, which is what the weeding literature uses (thin jaws, three-arm grippers, soil-loosening heads) rather than a stock Panda hand.
- **The arm grazes the bed during the twist.** link5 brushes it by 0.94 mm. Collision avoidance constrains joint velocity and is re-applied each control tick; the twist asks for a large reconfiguration in one tick, so the solve ends on the boundary instead of a full CLEARANCE clear of it.
- **Not every weed is covered by tests.**
  - The tests drive shepherd_1 end to end.
  - They check that sowthistle_1 reaches the grip still rooted, and that sowthistle_2 is held through the lift.
  - Every other weed has only been checked by hand, with the launcher or a one-off script.
  - shepherd_2 and shepherd_4 have not been re-run since the grasp tolerance was added.
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
- **M7 — Arm IK to a pre-grasp pose. DONE.** mink differential IK, collision-aware.
- **M8 — Grasp. DONE for the shepherd's purse and the sowthistle.** Descend through the foliage, close on the stalk, and confirm the hold through fingertip contact with the stem (both pads, or one pad with the other within 1.0 mm).
- **M9 — Pull and retreat. DONE for the shepherd's purse; partial for the sowthistle.** On the shepherd's purse, twist plus lift exceeds the combined root criterion; the plant separates and the arm returns home. On the sowthistle the root breaks on the twist alone (see §6).
- **M10 — Orchestrator.** State machine over all weeds, never grasping a strawberry.

## 8. Technical Constraints and Decisions

**Simulator:** MuJoCo 3.8.1, native Python bindings. Pinned to `.venv`.

**Solver:** `timestep=0.001`, `cone="elliptic"`, `impratio=10`, Newton. Elliptic cones and a high impratio are what MuJoCo's documentation prescribes to stop a gripped object sliding free.

**No air.** MuJoCo's fluid model cost 17% of the step and changed neither fall time (248 ms over 30 cm, against a 247 ms ideal) nor tumble. Removed. Joint damping is not an alternative either: one value covers all six DOFs of a free joint, and anything large enough to slow a fall dwarfs a plant's rotational inertia, so the plant stops tumbling and sinks upright.

**M1 base:** slide joint `base_slide_x`, axis `1 0 0`, range `-2 4` m, damping 50, armature 1, frictionloss 1; base mass 30 kg; `<position>` actuator kp 2000, kv 300, forcerange ±1500 N.

**Actuator order:** `actuator1..7` (arm), `actuator8` (gripper, ctrlrange 0–255), `base_x`.

**D435i:** `pos="0.05 0 0.04"` on the `hand` body, `xyaxes="0 -1 0 1 0 0"`. Colour camera at x=+0.0325, depth at x=+0.0175, both 4.2 mm behind the glass; fovy 42° and 58° per the datasheet.

**Weed collision:** visual mesh is `contype=0 conaffinity=0 group=2`; collision is the CoACD decomposition at `group=3 condim=4 friction="1.5 0.02 0.001" solref="0.005 1" solimp="0.05 0.99 0.012"`. The stem is compressible, and the *shape* of the give matters as much as the amount: impedance ramps from almost nothing at first touch to nearly rigid over 12 mm, so the pads sink in and then meet something firm enough to hold. Measured at the grip: 8.5 mm of squash at 9.3 N. A uniformly soft stem squashes but cannot be gripped -- the pads sink in without building force and the stem squirts out sideways, which capped the grip near 3 N however hard the fingers were commanded, and the extraction failed. On MuJoCo's rigid defaults the opposite happened: an immovable wall, 4.9 mm of interpenetration at 66.7 N, and the weed carried around on friction long after the fingers opened. MuJoCo replaces a mesh geom with its convex hull, which is 20–70 mm wide where the real stem is 1–2.5 mm — the fingers would close on empty air. Measured graspable width at the shepherd's stalk: 3.4 mm.

**Root anchorage:** one `<weld>` per weed, `anchor` at the planted position, `solref="0.01 1" solimp="0.95 0.99 0.001"`. `anchor` must be set explicitly; the default relpose drags the plant off its spot. Break force 15 N, sustained 20 ms, counted only along the planting axis. MuJoCo equalities have no break force, so `Roots.snap()` does it.

**Stalk springs:** hinge `stiffness=0.05`, `damping=8e-4`, `armature=1.4e-4`, `range=±0.5 rad`. The armature is what sets the sway — the segments' own inertia would put the stalk at 27 Hz, which reads as a twitch. The travel limit matters as much: at ±1.2 rad the plant folded flat and dropped its own tip 20 cm, which turned any sideways drag into an upward one.

**Plant masses:** sowthistle 15 g, shepherd's purse 8 g, set by explicit `<inertial>`. Mesh-derived inertia is not trustworthy — the meshes are not watertight.

**Interactive drag:** a critically damped spring sized to the grabbed body's mass, `ROOT_BREAK_FORCE / DRAG_TRAVEL` stiffness, clamped at 1.3×. MuJoCo's own perturbation scales with mass and gives a 15 g plant ~0.3 N. An undamped spring on a 2 g stalk tip rings at ~80 Hz and throws force spikes into the root.

**Root failure criterion:** tension and torsion combine linearly -- `axial/15 N + torsion/0.07 N.m >= 1`, sustained 20 ms. A root shears in torsion as well as pulling out, which is why twisting a weed loose beats hauling on it. `ROOT_BREAK_TORQUE` is 0.07 N.m in `env.py`. This section previously recorded 0.04 N.m, reasoned as follows: the same soil friction resists both pull and twist, acting at the root's own radius, so the torque it takes is the break force times that radius -- 15 N across a 2.7 mm taproot. The first value, 0.15, assumed a 10 mm radius and was far too generous; at 0.04 the twist genuinely carries the extraction, which is the point of twisting a weed rather than hauling on it. Measured: a sustained twist about the planting axis used to read exactly 0.00 N of axial force and never uprooted anything, because `axial_force` reads only the first three of the weld's six constraint rows.

**IK:** mink 1.2.0 -- the last release that accepts `mujoco==3.8.1`; 1.3.0 requires 3.10. Built on a plant-free copy of the scene, because the weeds' 96 free-joint DOFs would otherwise enter every QP as variables the arm cannot move (3.5x slower, and mink's posture and configuration limits divide by zero on them). The rail and the fingers are frozen with `DofFreezingTask` so an unreachable target stays unreachable. Joint velocity limits are mandatory, not cosmetic: collision avoidance constrains velocity, so without a speed cap the solver takes a step big enough to jump the 80 mm detection band and puts the elbow through a bed. `base_cart` is excluded from collision avoidance -- it is a visual box drawn under `link0`, so `link1` touches it at zero distance and any clearance against it makes the QP infeasible.

**Grasping:** a MuJoCo parallel jaw holds by friction, and this one cannot. The menagerie gripper servo is `biasprm="0 -100 -10"`, about 0.1 N per mm, so on a 3 mm stalk it squeezes ~1.3 N and holds ~2 N against a root that takes 15 N. Left to friction the jaws close straight through the stem and shut on each other -- measured finger gap 0.0 mm through a "grasp", with the plant's collision pieces sitting inside the closed fingers and the actuator reading 0.00 N. The apparent 67 N grip seen earlier was the solver pushing overlapping shapes apart, never the actuator, which is also why the weed clung to the pads for seconds after they opened. So the hold is made explicit with a weld, toggled at runtime exactly as the root is, which is the standard approach in MuJoCo manipulation work. `env.Grasp` owns it; the honesty lives in `Grasp.closed`, a geometric test on the fingertip pads rather than on the gripper command (see Grasp tolerance below).

**Grippable bit:** the fingertip pads carry `contype=2 conaffinity=2` and only the stalk capsule and the soil advertise that bit. The pads therefore pinch stems and stop at the bed, and pass through foliage. Without this a leaf 22 mm from the axis jams the jaws at a 14 mm half-gap, 20 mm short of the stem.

**Grasp tolerance (`Grasp.REACH` = 1.0 mm):** `Grasp.closed(data, weed)` returns True in either of two cases:
- **Both pads:** both fingertip bodies are in contact with the weed. This is the original rule.
- **One pad (fallback):** exactly one fingertip is in contact with *that weed's own stem capsule*, and the other fingertip's pads are within 1.0 mm of the capsule, measured with `mujoco.mj_geomDistance`.

Why the fallback exists:
- The two fingers are linked, so the first pad to land on the stem stops both.
- The stem is welded rigidly at the root and cannot give the fraction of a millimetre a real stem would to centre itself.
- At sowthistle_2 and sowthistle_3 the arm settles 0.23 mm off-centre across the jaw. That left the second pad 0.3-0.7 mm short however long the jaws squeezed, and the old both-pads rule never fired.

What keeps it honest:
- Contact with the hand, the camera, leaves, the bed or another plant never counts. Only the target plant's stem capsule does.
- `Uprooter.step` only calls `closed` once the gripper has been commanded to `SHUT`, so an open jaw never registers as a grasp.

**Approach exclusions:** `weeds.xml` has a `<contact>` block that stops `hand` and `d435i` colliding with every sowthistle body.
- **Scope:** the root body and `_s1`, `_s2`, `_s3` for sowthistle_1-4, 32 pairs in all.
- **Why every segment is listed:** MuJoCo excludes apply to the named body only, not its children. An exclude on `sowthistle_1` alone left the hand still hitting `sowthistle_1_s1`.
- **What it fixed:** without it the hand pressed sowthistle_1 over by 35 deg with about 15 N and tore the root out at 8.06 s, during "close in", before the grip began.
- **Left unchanged:** fingertip pads, stem capsules, bed contact, the root welds and the shepherd's purse.

**Aiming:** `uproot.stem_centre` returns a point on the weed body's own z axis. Anything derived from the collision pieces -- their vertex centroid, or the midpoint of their span -- lands out in the leaves, several mm off the stalk, which on a 3 mm stem is the difference between a grasp and closing on air. Both were tried and both miss.

**Twist angle:** 10 degrees. With a rigid grasp the arm turns the plant against a stiff root, so the torque climbs with the angle and anything past about 15 degrees shears the root on the twist alone -- at 20 degrees and above the reaction saturates near 0.073 N.m whatever is commanded. Those figures, and the split below, were measured on shepherd_1 when `GRASP_HEIGHT` was still 14 mm: at 10 degrees the twist carried 67% of the root's capacity and the pull supplied the other 57%. At the current 16 mm, shepherd_1 measured about 66% twist and 52% pull.

**No artificial colour for classification** (hard requirement). `rgba` may be used for rendering realism only.

**Coordinate convention:** MuJoCo world frame, +z up, base travel along +x.

## 9. Software / Environment

**`.venv`** (Python 3.12, the only venv):
- `mujoco` 3.8.1, `mink` 1.2.0, `gymnasium` 1.3.0, `numpy` 2.2.6, `opencv-python` 4.12.0.88, `draccus` 0.11.6 — all pinned in `pyproject.toml` to match the Docker image.
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

If a sourced ROS environment puts its Python packages on `PYTHONPATH`, pytest auto-loads a ROS plugin and crashes before collecting (`No module named 'lark'`). Run it as `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../../.venv/bin/python -m pytest` instead.

---

## Maintenance Protocol (ONGOING RULE)

After **every** completed implementation step, update this file **before** starting the next. Append a dated entry with: milestone completed, files changed, implementation details, tests performed and results, current status, next milestone, and any decisions or parameters introduced. Also update the numbered sections above so the top of the document reflects reality.

---

## 2026-09-13 — M7-M9: collision-aware IK, root torsion, scripted extraction

**Milestone:** M7 done; M8 and M9 done for the shepherd's purse.

**Files changed**
- `workspace/hardware/panda_description/mobile_panda.xml` — added `grasp_site` on the `hand` body at `0 0 0.1034` (the Franka TCP, between the closed fingertip pads). It is the only thing IK aims at; the robot had no sites at all before.
- `weed_extraction/env.py` — `ROOT_BREAK_TORQUE`, `Roots.torsion`, `Roots.load`; `snap` now tests the combined criterion instead of the axial force alone. `Roots._rows` factors out the weld's six constraint rows.
- `weed_extraction/ik.py` — new. `ik_model`, `pose`, `ArmIK`.
- `weed_extraction/uproot.py` — new. `stem_centre`, `Uprooter`, the seven-phase sequence.
- `weed_extraction/launch.py` — `uproot` mode; `control_hz` and `weed` config fields; `drag_force` lost its `stiffness` parameter, `parse_config` lost both of its, and `capture` now calls `make_env` instead of repeating it.
- `configs/uproot.yaml`, `run_scripts/test_up_root.sh`, `tests/test_ik.py`, `tests/test_uproot.py` — new.
- `pyproject.toml` — `mink==1.2.0`, and a `slow` pytest marker.

**Tests:** 30 pass, up from 14. The three marked `slow` drive the whole arm through an extraction.

**Measured**
- IK: 7 iterations, 0.003 mm, ~0.25 ms per solve on the plant-free model (nq=nv=10 against the scene's 114/106).
- Extraction: uproots at t=12.1 s, 16.9 N of pull and 0.061 N.m of twist, gripped at the moment it separates.
- Twist alone used to do nothing: 0.00 N of axial reading at every torque from 0.02 to 0.5 N.m.
- The shepherd's stem at 6-8 mm above the soil is a 2.8 x 2.6 mm column, 1.0 mm off the body origin, and the only collision piece in the fingertips' window. At 10 mm and above, leaf pieces reaching 15.9 mm take over.

**Decisions:** see §8 for the root failure criterion and the IK constraints. Three dead ends worth not repeating: `base_cart` cannot be given collision or any IK clearance (it overlaps `link1` by construction); the gripper needs a smaller clearance than the arm or it can never reach the soil; and without joint velocity limits the collision avoidance is bypassed entirely.

**Next:** M6 perception, then M10 orchestration. Before either, the sowthistle needs a graspable stem (see §6).

---

## 2026-09-13 (later) — side grasp, and the base standing still

**Files changed:** `weed_extraction/ik.py` (`pose` now takes an approach direction and builds the frame from it, turning `twist` about the world vertical via `SO3.from_z_radians`), `weed_extraction/uproot.py` (rewritten: phases are a data table, `TILT`, base commanded once), `tests/test_ik.py`, `tests/test_uproot.py`.

**Two bugs fixed.** The base was re-deriving its target every tick from the per-phase clock, which restarts at each phase, so it paced back and forth; it is now commanded once in `__init__` and its servo takes it there. And `stem_centre` took its height as a default argument bound at import, so any override was silently ignored.

**The side grasp.** `TILT` is the angle off the stalk's own axis; 60 deg is the default. Measured required tool-point height against angle: 0-75 deg all need only 15-18 mm, but 90 deg needs 52 mm because link6 swings down — see §6. The twist is about the world vertical, which is the only rotation `Roots.torsion` reads.

**Correction to the earlier entry:** I first reported a level grasp as blocked from about 45 deg onward. That was wrong — it came from testing a handful of angles with cold IK solves from the home pose, where the collision limit stalls for want of room to work around. Sweeping the actual geometry shows everything up to 75 deg is comfortable.

**Tests:** 31 pass. `test_the_arm_never_drives_into_the_beds_or_the_ground` asserts a bounded graze rather than zero contact, for the reason in §6.

---

## 2026-09-15 — sowthistle grasp: approach, grasp height, grasp tolerance

**Milestone:** M8 extended to the sowthistle. M9 is still partial for it (see §6).

**Files changed**
- `workspace/application/field_description/weeds.xml` — `<contact>` block excluding `hand` and `d435i` from all 16 sowthistle bodies.
- `weed_extraction/uproot.py` — `GRASP_HEIGHT` 0.014 -> 0.016.
- `weed_extraction/env.py` — `Grasp.REACH`, `Grasp.pads`, `Grasp.stems`, and the one-pad fallback in `Grasp.closed`.
- `tests/test_uproot.py` — three new test functions (four test cases), listed below.

### a. Approach collision fix
- **Problem:** the hand and wrist camera struck the sowthistle canopy during "close in". sowthistle_1 was pressed 35 deg over with about 15 N and its root tore out at 8.06 s, before the grip began.
- **Fix:** exclude `hand` and `d435i` against every sowthistle body and segment.
- **Result:** all four sowthistles reach the grip still rooted.

### b. Grasp registration fixes
- **Grasp height:**
  - The arm settles about 2 mm below its commanded tool point.
  - At 14 mm the fingertip pads landed on the bed at seven of the eight weeds, and the jaws stalled there.
  - At 16 mm the pads clear the bed at all eight.
  - shepherd_1 worked at 14 mm only because its IK solve stopped 3.3 mm high.
- **One-pad tolerance:**
  - At sowthistle_2 and sowthistle_3 only one pad reached the stem, so the both-pads rule never fired. See "Grasp tolerance" in §8.
  - Both now register as held and are carried out of the bed.

### c. Remaining limitation (not fixed)
- All four sowthistles break their roots during the twist at about 0.4–0.5 N of pull and 0.070-0.074 N.m of twist.
- The lift carries a plant whose root has already broken.
- Cause not yet investigated. See §6.

**Tests:** 36 pass.
- `test_the_approach_does_not_tear_out_a_sowthistle` — sowthistle_1 is still rooted when the grip begins.
- `test_one_pad_on_the_stem_is_a_grasp_only_if_the_other_is_close` — one pad touching and the other 0.5 mm away counts; 2.0 mm away does not.
- `test_a_stem_off_centre_in_the_jaws_is_still_held` — sowthistle_2 is held for the whole lift.

**Next:** find why the sowthistle's root reaches the twist limit after only a few degrees of turn.
