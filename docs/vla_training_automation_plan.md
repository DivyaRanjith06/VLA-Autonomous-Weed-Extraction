# SmolVLA Weed Extraction: Automated Training Plan

## Status

This document is the single source of truth for dataset collection, environment generation, episodes, validation, and training for the SmolVLA weed-removal task.

The stale collection note in [docs/smolvla_data_collection.md](smolvla_data_collection.md) is retired. All collection logic and task design live here.

## Goal

Train a SmolVLA policy that can follow natural-language prompts such as:

- remove all weeds
- remove the shepherd weed
- remove the sowthistle weed
- remove the strawberry from the left bed
- remove one specific weed from a species family
- remove the nearest weed
- remove all weeds of one species from a target area

The policy must work across many scenes, soil conditions, lighting conditions, crop layouts, and root strengths while not harming strawberry plants. The simulation must generate demonstrations with enough variation for robust generalization and enough metadata to diagnose failure modes.

## Guiding principle: collect the right data, not just more data

The SmolVLA docs recommend recording on the order of 50 episodes as a minimum starting point for a task, with variation repeated across positions and settings. For rookies, a small but diverse dataset is better than a large but narrow one. On Apple Silicon, use MPS and keep the dataset compact but varied.

For our weed task, the best collection strategy is:

- 20 scenes total, organized as 4 environment families × 5 variants
- 6–8 strawberry plants per scene
- weed counts that vary per scene, from sparse to dense
- species-aware instructions and prompts, not just a single “remove the weed” task
- per-frame streams for image, joint state, gripper state, end-effector state, and force/torque
- hidden metadata for root strength and scene randomization so we can audit policy failures without leaking those values to the model

## Scene generation: 15 scenes and the per-scene layout

Each scene must contain 6–8 strawberry plants and a randomized number of weeds. This is the minimum target set for a first dataset on a Mac M4.

### 15-scene plan

Create 15 scenes as five variants within three scene families:

1. Midday, dry soil
2. Midday, loose soil
3. Evening, warm light
4. Evening, cool light
5. Mixed clutter / dense vegetation

Each family gets 5 random draws, for a total of 15 scenes. This is better than 15 totally unrelated scenes because the dataset stays physically coherent while still covering variation.

### Per-scene layout policy

Each scene includes:

- 6–8 strawberry plants
- 3–9 weeds total
- weed species mix randomized, but balanced across the dataset
- species and position labels recorded in metadata
- crop and weed layouts generated with collision checks before rendering
- at least one valid target area for each prompt type

Weed count should vary by scene, for example:

- sparse scene: 3 weeds + 6 strawberries
- normal scene: 5 weeds + 7 strawberries
- dense scene: 8 weeds + 8 strawberries

The final dataset should not be biased to one weed count or one crop density.

## What to randomize

These are the exact randomization dimensions that define the dataset:

### 1) Scene and ground parameters

- bed height offset and local slope
- ground patch or texture selection
- soil dryness / roughness / compactness
- walking path offset and visual variation
- crop-bed width and row spacing
- small camera-to-bed translations

### 2) Plant placement

- weed x/y positions
- weed yaw / orientation
- small stem lean
- root depth and planting offset
- crop plant x/y positions
- crop plant yaw and leaf spread
- local weed-to-crop spacing
- plant occlusion level

### 3) Crop layout for strawberries

- 6–8 strawberry plants per scene
- randomized plant pose and maturity per scene
- yield variation by local leaf density
- orientation sampled over yaw with minimum angular separation
- ensure no exact overlap with the weed target region

### 4) Weed family and species composition

- shepherd
- sowthistle
- mixed species scenes
- single-species scenes
- species-specific prompts

### 5) Lighting

- midday bright and neutral
- evening warm
- evening cool
- low-light with more shadow
- random exposure shifts

### 6) Root / physics variation

- root break force per weed around the current 15 N regime
- root break torque per weed tied to root radius and stiffness
- stalk stiffness/damping variation
- friction and soil resistance variation
- random drag/twist variation during extraction

### 7) Prompt / task-language variation

- remove all weeds
- remove the shepherd weed
- remove the sowthistle weed
- remove one specific weed by index/ID
- remove one weed from a species family only
- remove all weeds of one family/species
- remove one weed from the left/right bed
- remove the nearest target weed

This prompt set is essential for training language grounding. Every episode must record the language instruction used.

## Prompt taxonomy for the final application

The policy should be trained to handle all of these task types:

1. All weeds

   - “remove all weeds”
   - “remove every weed in the bed”
2. One weed by species

   - “remove the shepherd weed”
   - “remove the sowthistle weed”
3. One specific weed instance

   - “remove weed 3”
   - “remove the weed on the right side of the bed”
   - “remove the nearest weed”
4. One species family only

   - “remove all shepherd weeds”
   - “remove all sowthistle weeds”
5. One weed from one family / species / region

   - “remove one shepherd weed from the left bed”
   - “remove one sowthistle weed near the front row”
6. Crop-safe task variants

   - “remove the weed but do not touch the strawberries”
   - “remove only the weeds, leave the crop untouched”

For training, keep language templates canonical and deterministic, but randomize the object references and scene layout. Do not expose simulator metadata or internal root strength to the model.

## Root-force logic and extraction condition

The current root break logic is already the right abstraction: the weed releases when the combined load exceeds the failure criterion for a sustained period.

The condition is:

```text
load = pull / ROOT_BREAK_FORCE + twist / ROOT_BREAK_TORQUE
if load > 1.0 for >= 0.02 s, then the root gives way
```

This matters because a plant can uproot under combined tension + twist even when the pure pull reading alone is below 15 N. In practice the data can show values such as:

- pull ≈ 7.8 N
- twist ≈ 0.047 N·m
- combined load > 1.0

That is why the root breaks under combined loading, not pure pull-only logic.

For data collection we should randomize around the nominal 15 N behavior while keeping the same combined-load physics. The policy must learn from the observed response, not from the hidden root-strength variable.

## Required data streams for SmolVLA training

The dataset must contain the following streams at every timestep or every control step:

### Camera streams

- wrist RGB camera
- wrist depth camera (if available)
- overview RGB camera (helpful for scene context and task grounding)
- optional second overview camera or side camera if the simulator supports it

Recommended camera frame format:

- RGB: uint8, 3-channel
- depth: float32 or normalized depth map
- if storage is tight: keep wrist RGB + overview RGB with optional depth only on key steps

### Robot state streams

- joint positions (qpos)
- joint velocities (qvel)
- end-effector pose
- gripper state or width
- base rail position
- actuator command values

### Force / tactile streams

- measured wrist force
- measured wrist torque
- contact or collision counts
- contact body IDs when relevant
- grasp state and hold state

### Task / episode metadata

- episode seed
- scene family ID
- scene variant ID
- prompt string
- target weed species
- target weed instance ID
- crop and weed counts
- root strength parameters used
- success/failure reason
- phase labels: reach, settle, close in, grip, twist, lift, carry, drop

### Label streams

- target-identified flag
- grasped flag
- uprooted flag
- carried flag
- released-over-bin flag
- crop-damage flag
- failure reason

### Action streams

Use the current 9D actuator-target action as the first dataset contract:

- 7 joint positions
- 1 gripper command
- 1 base x target

This matches the simulator and keeps the dataset faithful to the current control loop. A later policy contract can convert to delta pose + gripper, but do not mix both in one dataset.

## Episode contract and collection order

Each episode must record data in a causal order:

```python
observation = env.get_obs()
phase = expert.step()
action = env.data.ctrl.copy()
next_observation, _, terminated, truncated, info = env.step(action)
```

Record the observation that caused the action, not only the post-step observation. Every accepted frame should include:

- image frame(s)
- joint positions
- joint velocities
- gripper state
- robot state
- action taken
- phase transitions
- metadata and event flags

## Validation filters for accepted episodes

Accept an episode only when all of the following are true:

- target was reached and grasped before lift
- target root released under the combined-load criterion
- extraction happened during twist/lift, not during approach collision
- target crop was not damaged
- weed was carried to the tray and released there
- the simulation remained finite and stable
- image, state, and action lengths match exactly

Reject or quarantine episodes when:

- target is fully occluded
- target is only identifiable by fixed texture or location
- species is missing from a soil family
- crop and weed layouts are too similar across scenes
- yaw pattern repeats too closely
- root thresholds or scene data are inconsistent

Keep rejected episodes in a `rejected.jsonl` manifest for auditing but exclude them from the training set.

## 15-scene sampling logic

Use a structured sampling strategy rather than fully independent randomization:

- 3 scene families: midday, evening, mixed clutter
- 5 variants per family
- each variant randomly samples:
  - ground height and soil type
  - lighting preset
  - crop placement and orientation
  - weed count and species mix
  - root strength per weed
  - prompt label and target ID

This preserves realistic combinations while creating enough diversity to generalize.

## Data collection best practice for this project

The best approach is not to generate arbitrary “more frames.” The best approach is to generate structured, labeled, task-aware demonstrations with broad but coherent variation.

Recommended first dataset target:

- 50 to 100 episodes as the first pass
- 15 scenes, with repeated trials per scene
- 5–10 successful demonstrations per scene variant
- balanced prompt coverage across the prompt taxonomy
- at least 10 examples per species and per task type where possible

The official SmolVLA docs note that around 50 episodes across several variation points is a good starting point, and that more varied demonstrations generalize better than a single long perfect run. That is the right starting point for this project.

## M4 Mac training reality check

For Apple Silicon M4 / MPS training, use the smallest practical configuration first:

- `policy.device=mps`
- `batch_size=1` or `2`
- reduced image size, e.g. 320×240 or 480×360
- `num_workers=0` or very low
- `freeze_vision_encoder=True` at first if memory or speed is tight
- start with 10k–20k steps, not a huge run

This is the practical M4 path because SmolVLA is still a relatively heavy model and Apple Silicon training is slower than NVIDIA GPUs.

Recommended first command:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=USER/WEED_DATASET \
  --policy.device=mps \
  --policy.dtype=float32 \
  --batch_size=1 \
  --num_workers=0 \
  --steps=10000 \
  --output_dir=outputs/train/smolvla_m4
```

If the training is stable and the dataset quality is good, move to 15k–20k steps. Do not start with a large batch or large images on M4 hardware.

## Recommended data run directory

```text
outputs/
  generation/
    run_2026_09_20/
      run.json
      manifest.jsonl
      rejected.jsonl
      episodes/
        episode_00001/
          observations.npz
          frames/
          episode.json
```

Each episode record should include:

```json
{
  "episode_index": 12,
  "seed": 48192,
  "task": "remove all weeds",
  "scene_family": "midday_dry",
  "scene_variant": 3,
  "target": "shepherd_2",
  "weed_count": 6,
  "strawberry_count": 8,
  "success": true,
  "peak_pull_n": 14.2,
  "peak_twist_nm": 0.041,
  "root_break_force_n": 15.8,
  "root_break_torque_nm": 0.062,
  "frames": 318,
  "fps": 20
}
```

## Final implementation policy

This project should use the following sequence:

1. Generate 15 scene variants with 6–8 strawberries each.
2. Randomize weed counts and species mix per scene.
3. Generate prompt-aware episodes for all task families.
4. Collect full camera + state + action + metadata streams.
5. Validate episodes, reject bad ones, and keep only accepted demonstrations.
6. Train on the accepted dataset with MPS-compatible settings on M4.
7. Evaluate on held-out scene combinations and prompt families.

This is the dataset that should be used for the first SmolVLA fine-tuning pass.

## Summary

The key design changes for this project are:

- 15 scene families with 6–8 strawberries per scene
- randomised weed count, species mix, location, orientation, and root strength
- prompt-based tasks for all the user intents the application must support
- full data capture for cameras, joint states, actions, and force/torque
- MPS-friendly training settings for an M4 Mac
- single-source-of-truth documentation in this file

Start with one RGB camera and frozen vision/action-expert settings supported by the installed LeRobot version. Use gradient accumulation before increasing MPS batch size. Keep `PYTORCH_ENABLE_MPS_FALLBACK=1` only when an operator requires it because CPU fallback can make training very slow. Do not assume CUDA AMP, FlashAttention, `torch.compile`, bitsandbytes quantization, or MLX conversion works for SmolVLA on MPS.

The M4 is a good machine for simulator debugging, dataset validation, small fine-tunes, and inference. Use an NVIDIA GPU or cloud instance for larger sweeps once the data pipeline is proven.

## Recommended Implementation Phases

1. **Parameterize mechanics**: move root force/torque from module constants into reset-time environment state; add tests for 15, 17.5, and 20 N cases.
2. **Add deterministic sampler**: implement the six variation families, rejection checks, manifests, and a test that identical seeds produce identical samples.
3. **Add recorder**: record causal observation/action pairs, phase events, root wrench, and hidden metadata for one scripted episode.
4. **Add recovery tree**: implement bounded grasp, extraction, and tray-drop retries; preserve failure labels.
5. **Add parallel generation**: process-isolated workers and artifact validation; generate a small smoke dataset first.
6. **Pack and inspect**: convert accepted artifacts to LeRobot Dataset v3 and verify feature shapes, frame counts, and statistics.
7. **Train baseline**: fine-tune SmolVLA on nominal plus pose/lighting families, then add dynamics and combined families.
8. **Evaluate generalization**: run the held-out family and publish factor-wise full-task success, extraction force, tray-drop rate, and crop-safety rate.

## Acceptance Criteria

The automation is ready for a first training run when:

- at least six variation families generate valid scenes;
- 100 consecutive seeded episodes produce reproducible manifests;
- the expert completes extraction and tray drop for the nominal family;
- root force values cover 15-20 N and are hidden from policy observations;
- each soil family contains both weed species and strawberry neighbors, including brown-mud texture variants, without species-level soil-color shortcuts;
- strawberry yaw sampling passes the within-scene and cross-scene near-duplicate rejection checks;
- failed episodes are classified and recoverable without corrupting artifacts;
- the LeRobot dataset passes frame/action/statistics validation;
- per-class held-out metrics cover weed-vs-weed, weed-vs-strawberry, low-color-contrast, and temporal-view cases;
- a held-out family demonstrates whether the policy generalizes instead of merely replaying one scene.

## References

- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [LeRobot SmolVLA guide](https://huggingface.co/docs/lerobot/en/smolvla)
- [LeRobot Dataset v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [LeRobot repository](https://github.com/huggingface/lerobot)
- [robosuite](https://github.com/ARISE-Initiative/robosuite)
- [MimicGen](https://github.com/NVlabs/mimicgen)
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [RoboCasa](https://github.com/robocasa/robocasa)
- [OpenVLA](https://github.com/openvla/openvla)
- [NumPy parallel random generation](https://numpy.org/doc/stable/reference/random/parallel.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
