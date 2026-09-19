# SmolVLA Weed-Picking Data Collection

This document describes the data pipeline needed to train SmolVLA to pick weeds whose root resistance changes from episode to episode.

The design does not depend on the vendored `lerobot` folder. The simulator writes a simple, explicit dataset, and a separate SmolVLA adapter reads that dataset. This keeps the simulator independent from a training framework.

The important rule is simple:

> The root strength is randomized in simulation, but the policy never receives the root strength directly. It must adapt from what it sees and feels.

## 1. Training Architecture

```text
+--------------------------+
| MuJoCo field scene       |
| robot, weeds, crop, soil |
+------------+-------------+
             |
             | reset episode
             v
+--------------------------+
| Episode randomizer       |
| - weed species/location  |
| - root force             |
| - root torque            |
| - camera and physics     |
+------------+-------------+
             |
             | hidden simulator state
             v
+--------------------------+
| MuJoCo simulation        |
| robot + root constraint  |
+------+-------------------+
       |
       | expert or policy action
       v
+--------------------------+
| Action adapter            |
| delta pose + gripper      |
|        ->                 |
| target pose + IK          |
|        ->                 |
| joint actuator targets    |
+------+-------------------+
       |
       v
+--------------------------+
| Robot motion              |
| Panda arm, gripper, rail  |
+------+-------------------+
       |
       +------------------------------+
       |                              |
       | next observation             | success and physics data
       v                              v
+--------------------------+   +--------------------------+
| Policy observation       |   | Episode recorder         |
| - RGB image              |   | - action                 |
| - joint state            |   | - reward/events          |
| - end-effector state     |   | - force/torque           |
| - wrist force/torque     |   | - hidden metadata        |
+------------+-------------+   +------------+-------------+
             |                              |
             v                              v
       SmolVLA input                    SmolVLA dataset
             |                              |
             +--------------+---------------+
                            v
                       SmolVLA trainer
```

## 2. What Happens in One Episode

```text
1. Reset MuJoCo.
       |
2. Choose weed species and location.
       |
3. Randomize root resistance.
       |  Hidden from the policy.
       |
4. Render RGB and collect robot/force state.
       |
5. Choose an action.
       |  During data collection: scripted Uprooter expert.
       |  During evaluation: SmolVLA policy.
       |
6. Convert action into safe joint targets with IK.
       |
7. Step MuJoCo physics.
       |
8. Record observation, action, event, and outcome.
       |
9. Repeat until the weed is placed, the episode fails, or it times out.
```

The current scripted path starts here:

```text
run_scripts/test_up_root.sh
    -> weed-launch uproot
    -> launch.py
    -> Uprooter
    -> ArmIK
    -> MuJoCo
```

`Uprooter` is useful as an expert demonstration generator. It should not be the policy itself.

The recorder should write ordinary image files plus structured episode files such as JSONL, NumPy, or Parquet. A SmolVLA data adapter can convert those records into the exact tensor format required by the installed SmolVLA implementation.

## 3. Data Seen by the Policy

Each training step should contain the following observation:

```text
observation_t = {
    image:              wrist RGB image,
    joint_position:     7 arm joint positions,
    joint_velocity:     7 arm joint velocities,
    gripper_position:   gripper state,
    base_position:      rail position,
    end_effector_pose:  position and orientation,
    wrist_force_torque: Fx, Fy, Fz, Tx, Ty, Tz,
}
```

The policy may also receive a language instruction such as:

```text
"Pick the weed and place it in the bin."
```

The policy must **not** receive:

```text
root_break_force
root_break_torque
root type from simulator metadata
MuJoCo equality or constraint identifiers
future state
```

Those values belong in private episode metadata for analysis only.

## 4. Policy Action

Use a task-level action instead of asking SmolVLA to learn seven independent joint trajectories:

```text
action_t = {
    dx, dy, dz,
    droll, dpitch, dyaw,
    gripper,
}
```

The action flow is:

```text
SmolVLA delta action
        |
        v
current end-effector pose + delta pose
        |
        v
Mink pose target
        |
        v
ArmIK.solve()
        |
        v
seven arm joint targets + gripper command
        |
        v
MuJoCo actuator controls
```

This keeps IK responsible for robot geometry and safety while SmolVLA learns the task:

```text
SmolVLA decides:      where and how to move next
IK decides:           which joints can safely make that move
MuJoCo decides:       what physical force results
```

## 5. Root-Strength Randomization

The current implementation uses one global value:

```python
ROOT_BREAK_FORCE = 15.0
ROOT_BREAK_TORQUE = 0.07
```

For training, move these values into episode state:

```text
weak root:    10 N
normal root:  15 N
strong root:  20 N
```

A physically consistent version derives torque resistance from force and root radius:

```text
root_break_torque = root_break_force * root_radius
```

At reset:

```text
randomize root parameters
reset root overload timers
reset robot and weed poses
```

The root model then evaluates:

```text
load = measured_pull / episode_break_force
     + measured_twist / episode_break_torque
```

The weed is released only when the load stays above the failure threshold for the required duration.

## 6. Force and Torque Recording

For simulation training, record the reaction at the wrist or grasp point:

```text
Fx, Fy, Fz: force in newtons
Tx, Ty, Tz: torque in newton-metres
```

For a future real robot, use the actual wrist force/torque sensor with the same observation format.

Do not train on the hidden root threshold. Train on the measured response:

```text
pulling harder -> measured force rises
root releases  -> force response changes
weed is loose  -> lift and carry
```

This is what allows the policy to handle different weeds with the same appearance but different resistance.

## 7. What to Record for Every Step

Store one record for every control tick:

```text
{
    "episode_index": 42,
    "timestamp": 3.20,
    "observation": {
        "image": "frames/0042/000160.png",
        "joint_position": [...],
        "joint_velocity": [...],
        "end_effector_pose": [...],
        "wrist_force_torque": [...],
        "gripper": ...,
        "base_position": ...
    },
    "action": {
        "delta_pose": [...],
        "gripper": ...
    },
    "event": "lift",
    "done": false
}
```

Keep these fields in episode metadata, but do not place them in the policy observation:

```text
{
    "weed_species": "shepherd",
    "weed_name": "shepherd_1",
    "root_break_force": 18.4,
    "root_break_torque": 0.08,
    "random_seed": 1234,
    "success": true,
    "failure_reason": null,
    "peak_force": ...,
    "peak_torque": ...,
    "crop_contact": false
}
```

## 8. Demonstration Collection

Use the scripted controller first:

```text
Uprooter
    -> generates reach, grasp, twist, lift, carry, and drop actions
    -> ArmIK converts each target pose to joint targets
    -> recorder stores policy-format observations and delta-pose actions
```

For every demonstration:

- Randomize the root strength.
- Use several weed species and positions.
- Include successful and failed attempts.
- Record measured force and torque.
- Store the expert action as a delta pose, not only as seven joint angles.
- Mark the exact time of grasp, root release, lift, and drop.

The expert should be changed later so it can adapt its pull and twist to measured force. A fixed expert that always uses the same motion will produce weak demonstrations for strong roots.

## 9. Recommended Dataset Splits

Train with:

```text
weed locations:      80% of locations
root strengths:      10-18 N
camera variations:   training range
```

Validate with:

```text
new episodes from known species and locations
root strengths inside the training range
```

Test with:

```text
held-out locations
held-out root strengths, for example 9 N, 19 N, and 20 N
both weed species
```

Do not split frames randomly across train and test. Split by episode. Otherwise frames from the same extraction sequence can appear in both sets and make results look better than they are.

## 10. Success Criteria

A successful episode must satisfy all of these:

```text
weed grasped
root released
weed lifted without crop damage
weed carried to the bin
weed released in the bin
```

Track these metrics:

```text
reach success
stable grasp success
root extraction success
complete placement success
crop contact rate
peak force
peak torque
time to extraction
success by root strength
success by weed species
```

## 11. Implementation Order

```text
1. Make root force and torque episode-specific.
2. Add wrist force/torque to observations.
3. Add delta-pose action conversion through existing IK.
4. Add task events, rewards, and termination conditions.
5. Add a standalone episode recorder and SmolVLA data adapter.
6. Generate demonstrations with Uprooter.
7. Train SmolVLA on RGB, state, and force/torque.
8. Evaluate on unseen root strengths and locations.
```

The existing IK should remain in the control stack. It is not unnecessary duplicated work: it is the safety and robot-geometry layer between the VLA action and MuJoCo.
