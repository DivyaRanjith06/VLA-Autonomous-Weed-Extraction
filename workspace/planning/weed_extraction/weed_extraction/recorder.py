"""Level 3 foundation: records one expert demonstration to disk.

Follows the plan's causal contract exactly, once per control tick:

    observation = env.get_obs()
    phase = expert.step()
    action = env.data.ctrl.copy()
    next_observation, reward, terminated, truncated, info = env.step(action)

`observation` -- images, joint state, end-effector pose, gripper width, base x, and
the target's root wrench -- is read before `expert.step()` touches anything, so every
recorded field at index i is the state that produced `action[i]`, never the state that
resulted from it. Grasp state (`held`) is part of that same snapshot: the frame where
the gripper closes on the stem still records `held=False`, since that was true when the
action was chosen: `Grasp.closed()` is that step's expert, deciding from the previous
step's contacts.

Failed attempts stay in the episode. The recovery tree (see weed_extraction.recovery)
judges each tick's snapshot before the expert acts; on a failure it starts the next
attempt, whose first action is that tick's, so the frame belongs to the new attempt. The
`attempt` stream labels every frame and episode.json holds each attempt's outcome.

Multi-weed prompts ("remove all weeds") are out of scope here: `Uprooter` drives one
weed per run, so one episode is one weed. A prompt naming more than one is refused.
"""

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np

from weed_extraction.ik import ArmIK
from weed_extraction.recovery import Recovery, RecoveryPolicy

# Every per-frame stream in observations.npz besides the cameras. Module level because
# the validator imports it to know what a complete episode holds.
STREAMS = ("phase", "attempt", "action", "uprooted_this_step", "held", "ee_pos", "ee_quat",
           "gripper_width", "base_x", "pull_n", "twist_nm", "crop_contact_n", "qpos", "qvel")


def ee_pose(model, data):
    """World-frame position and (w, x, y, z) orientation of `grasp_site`."""
    site = model.site("grasp_site").id
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[site])
    return data.site_xpos[site].copy(), quat


def gripper_width(model, data):
    """Metres between the fingertips: the linked fingers' joint positions sum to it."""
    fingers = [model.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")]
    return float(data.qpos[fingers].sum())


def base_x(model, data):
    return float(data.qpos[model.joint("base_slide_x").qposadr[0]])


def target_wrench(env, root_eq):
    """Pull (N) and twist (N.m) on the target weed's root, or (0, 0) once it is loose."""
    if not env.data.eq_active[root_eq]:
        return 0.0, 0.0
    return env.roots.wrench(env.data, root_eq)


def crop_contact(model, data):
    """Largest normal force, in newtons, on any strawberry right now, and each
    (strawberry, other plant or "mobile_base" for the robot) pair touching.

    Strawberries are fixed bodies with no damage model, so contact is all the simulator
    can say about them: a touch is measured, harm is not.
    """
    worst, pairs, force = 0.0, set(), np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.dist > 0:
            continue
        bodies = [model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]]
        crops = [b for b in bodies if model.body(b).name.startswith("strawberry")]
        if not crops:
            continue
        other = next(b for b in bodies if b != crops[0])
        mujoco.mj_contactForce(model, data, i, force)
        worst = max(worst, abs(float(force[0])))
        pairs.add((model.body(crops[0]).name, model.body(model.body_rootid[other]).name))
    return worst, pairs


def bin_interior(model, data):
    """The bin's inside as a world-frame box: the floor's footprint, from the floor's
    top to the walls' top. The bin only slides with the base, so the box stays square
    to the world axes."""
    floor, wall = model.geom("bin_floor").id, model.geom("bin_wall_xp").id
    centre, half = data.geom_xpos[floor], model.geom_size[floor]
    top = data.geom_xpos[wall][2] + model.geom_size[wall][2]
    return ([centre[0] - half[0], centre[1] - half[1], centre[2] + half[2]],
            [centre[0] + half[0], centre[1] + half[1], top])


def record_episode(env, manifest, prompt, out_dir, ik=None, policy=RecoveryPolicy()):
    """Run the expert on `prompt`'s one target, retrying as `policy` allows, and save
    the episode under `out_dir`.

    Writes `out_dir/observations.npz` (every per-step stream, stacked) and
    `out_dir/episode.json` (scene and episode metadata). Returns `out_dir`.
    """
    if len(prompt.targets) != 1:
        raise ValueError(f"one episode drives one weed; prompt names {prompt.targets}")
    target = prompt.targets[0]
    root_eq = env.roots.of_body[env.model.body(target).id]
    control_hz = round(1 / (env.n_substeps * env.model.opt.timestep))

    tree = Recovery(env, target, ik or ArmIK(env.scene), policy)
    phase_order = [name for name, _, _ in tree.runner.phases]
    streams = {key: [] for key in STREAMS}
    images = {cam: [] for cam in env.cameras}
    uprooted_step = None
    uprooted = {}  # every weed that came loose, target or not, and the step it did
    crop_pairs = {}  # (strawberry, other) -> first step they touched

    while True:
        observation = env.get_obs()
        pos, quat = ee_pose(env.model, env.data)
        pull, twist = target_wrench(env, root_eq)
        held = env.grasp.held is not None
        crop_n, touching = crop_contact(env.model, env.data)

        i = len(streams["phase"])
        if (failed := tree.failure()) and not tree.fail(*failed, frame=i):
            break
        phase = tree.runner.step()
        if phase is None:
            landed = in_bin(env.data.xpos[env.model.body(target).id],
                            *bin_interior(env.model, env.data))
            tree.finish(i, landed)
            break
        action = env.data.ctrl.copy()
        _, _, terminated, truncated, info = env.step(action)

        uprooted_now = target in info["uprooted"]
        if uprooted_now and uprooted_step is None:
            uprooted_step = i
        for weed in info["uprooted"]:
            uprooted.setdefault(weed, i)
        for pair in touching:
            crop_pairs.setdefault(pair, i)

        streams["phase"].append(phase)
        streams["attempt"].append(tree.current.index)
        streams["action"].append(action)
        streams["uprooted_this_step"].append(uprooted_now)
        streams["held"].append(held)
        streams["ee_pos"].append(pos)
        streams["ee_quat"].append(quat)
        streams["gripper_width"].append(gripper_width(env.model, env.data))
        streams["base_x"].append(base_x(env.model, env.data))
        streams["pull_n"].append(pull)
        streams["twist_nm"].append(twist)
        streams["crop_contact_n"].append(crop_n)
        streams["qpos"].append(observation["qpos"])
        streams["qvel"].append(observation["qvel"])
        for cam in env.cameras:
            images[cam].append(observation[cam])

        # Past the end of the last attempt's sequence the next tick only closes it, so
        # running out of steps is a fault only while the expert still has phases to run.
        runner = tree.runner
        if (terminated or truncated) and runner.phase < len(runner.phases):
            full = sum(round(duration / runner.dt) for _, duration, _ in runner.phases)
            raise RuntimeError(
                f"env truncated at {i} steps before the expert finished (phase {phase!r}); "
                f"raise WeedExtractionEnv(max_steps=...) -- with retries an episode can "
                f"run {policy.max_attempts} sequences, up to {policy.max_attempts * full}")

    # The state the last action left behind: the terminal observation, which no frame
    # holds, since each frame records the state before its own action.
    lo, hi = bin_interior(env.model, env.data)
    final = {"target_pos": env.data.xpos[env.model.body(target).id].tolist(),
             "bin_interior": {"lo": lo, "hi": hi},
             "held": env.grasp.held is not None,
             "target_rooted": bool(env.data.eq_active[root_eq])}
    warnings = {mujoco.mjtWarning(w).name: int(env.data.warning[w].number)
                for w in range(len(env.data.warning)) if env.data.warning[w].number}
    return _save(env, manifest, prompt, target, out_dir, control_hz, streams, images,
                uprooted_step, phase_order, tree, policy, uprooted, crop_pairs, final,
                warnings)


def in_bin(point, lo, hi):
    """Whether `point` lies in the bin box `bin_interior` gives.

    Up to 5 mm below the floor's top still counts. A weed resting on the floor sinks a
    fraction of a millimetre into it through soft contact, and the floor is 15 mm thick,
    so a point that low is in the floor slab, not beneath the bin; one that missed lands
    on the walkway, 25 mm below.
    """
    sink = 0.005
    return all(low <= v <= high for v, low, high in
               zip(point, (lo[0], lo[1], lo[2] - sink), hi))


def _save(env, manifest, prompt, target, out_dir, control_hz, streams, images,
         uprooted_step, phase_order, tree, policy, uprooted, crop_pairs, final, warnings):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {k: np.stack(v) for k, v in streams.items() if k != "phase"}
    arrays["phase"] = np.array(streams["phase"], dtype="<U8")
    arrays["final_qpos"] = env.data.qpos.copy()
    arrays["final_qvel"] = env.data.qvel.copy()
    for cam, frames in images.items():
        arrays[cam] = np.stack(frames)
    np.savez_compressed(out_dir / "observations.npz", **arrays)

    root_eq = env.roots.of_body[env.model.body(target).id]
    attempts = tree.attempts
    final_outcome = attempts[-1].outcome
    success = final_outcome == "success"
    # The grasp of the last attempt. `held` is observed before each action, so a grasp
    # taken during frame i's action first shows at i+1, and belongs to phase i.
    frames = range(attempts[-1].first_frame + 1, len(streams["phase"]))
    grasp = next((i for i in frames if streams["held"][i] and not streams["held"][i - 1]),
                 None)
    grasped_at_phase = streams["phase"][grasp - 1] if grasp is not None else None
    episode = {
        "prompt": prompt.text,
        "prompt_kind": prompt.kind,
        "target": target,
        "scene": str(env.scene),
        "seed": manifest.seed,
        "family": manifest.family,
        "variant": manifest.variant,
        "split": manifest.split,
        "weed_count": len(manifest.weeds),
        "strawberry_count": len(manifest.strawberries),
        # What the root was actually given at reset, which a caller may have overridden;
        # the validator holds it to the manifest's strength.
        "root_break_force_n": env.roots.break_force * env.roots.strength[root_eq],
        "root_break_torque_nm": env.roots.break_torque * env.roots.strength[root_eq],
        "success": success,
        "final_outcome": final_outcome,
        "recovery": ("unrecovered" if not success
                     else "recovered" if len(attempts) > 1 else "none"),
        "attempt_count": len(attempts),
        "failure_labels": [a.outcome for a in attempts if a.outcome != "success"],
        "attempts": [dataclasses.asdict(a) for a in attempts],
        "recovery_policy": dataclasses.asdict(policy),
        "grasped": grasped_at_phase is not None,
        "grasped_at_phase": grasped_at_phase,
        # A grasp taken once the lift has begun is a coincidence, not a reach for it.
        "grasped_before_lift": (grasped_at_phase is not None
                                and phase_order.index(grasped_at_phase)
                                < phase_order.index("lift")),
        "uprooted_step": uprooted_step,
        "uprooted_phase": (streams["phase"][uprooted_step]
                           if uprooted_step is not None else None),
        "peak_pull_n": max(streams["pull_n"], default=0.0),
        "peak_twist_nm": max(streams["twist_nm"], default=0.0),
        "peak_load": attempts[-1].peak_load,
        "frames": len(streams["phase"]),
        "control_hz": control_hz,
        "cameras": list(env.cameras),
        "image_size": [env.height, env.width],
        "phases": phase_order,
        "uprooted": uprooted,
        "crop_contacts": [{"strawberry": s, "other": o, "first_step": step}
                          for (s, o), step in sorted(crop_pairs.items())],
        "final": final,
        "sim_warnings": warnings,
    }
    (out_dir / "episode.json").write_text(json.dumps(episode, indent=2))
    manifest.save(out_dir / "manifest.json")
    return out_dir
