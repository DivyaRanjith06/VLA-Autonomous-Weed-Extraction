# SmolVLA Weed Data Collection

This file has been retired as a source of truth.

The canonical project plan now lives in [docs/vla_training_automation_plan.md](vla_training_automation_plan.md).

All data collection, environment generation, randomization, dataset validation, prompt design, and M4/macOS training guidance should be implemented from that document.

This file is intentionally left as a redirect note so the repo does not point to stale collection guidance.

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
