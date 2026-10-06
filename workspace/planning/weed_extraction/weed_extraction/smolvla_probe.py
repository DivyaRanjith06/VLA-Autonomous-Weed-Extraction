"""Read-only check that the fine-tuned SmolVLA produces the robot's 9-D action.

    python -m weed_extraction.smolvla_probe [--prompt "remove the shepherd weed"]

Builds one real observation in a catalogue scene, with the robot at its home pose and in
the form the dataset stores it (pack.Layout), runs one SmolVLA inference, and prints and
saves the first action of the predicted chunk. The action is never sent to the robot:
this checks the interface, it is not closed-loop control, and it says nothing about how
well the policy does the task.
"""

import os

# Before LeRobot and transformers load: the checkpoint and tokenizer are read from the
# local cache, never fetched.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from weed_extraction.pack import Layout  # noqa: E402  (also puts LeRobot on the path)
from weed_extraction.recorder import ee_pose, gripper_width  # noqa: E402
from weed_extraction.scene import Manifest, build_env  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


def observe(config, manifest, scene_dir):
    """The policy's inputs at the home pose of `manifest`'s scene, and the cameras used."""
    cameras = {key: cam for cam, key in Layout.CAMERAS.items() if key in config.input_features}
    if any(cam.endswith("_depth") for cam in cameras.values()):
        raise SystemExit("this probe feeds colour cameras only")
    _, height, width = config.input_features[next(iter(cameras))].shape
    env = build_env(manifest, scene_dir, cameras=tuple(cameras.values()), width=width,
                    height=height, control_hz=25)
    obs, _ = env.reset()
    model, data = env.model, env.data
    position, quat = ee_pose(model, data)
    state = np.concatenate([
        data.qpos[[model.joint(j).qposadr[0] for j in Layout.JOINTS]],
        data.qvel[[model.joint(j).dofadr[0] for j in Layout.JOINTS]],
        position, quat, [gripper_width(model, data)]]).astype(np.float32)
    batch = {"observation.state": torch.from_numpy(state)}
    for key, cam in cameras.items():
        image = np.ascontiguousarray(obs[cam])
        batch[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    env.close()
    return batch, cameras


def probe(model_dir, prompt, manifest, scene_dir, device=None):
    """One inference; returns what it saw and the 9-D action it produced."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = device
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=config).eval()
    override = {"device_processor": {"device": device}}
    pre, post = make_pre_post_processors(config, pretrained_path=str(model_dir),
                                         preprocessor_overrides=override,
                                         postprocessor_overrides=override)
    batch, cameras = observe(config, manifest, scene_dir)
    batch["task"] = prompt
    torch.manual_seed(0)  # the action expert starts from noise
    began = time.time()
    with torch.inference_mode():
        action = post(policy.select_action(pre(batch))).cpu().numpy()[0]
    seconds = time.time() - began
    if action.shape != (len(Layout.ACTIONS),):
        raise AssertionError(f"expected a {len(Layout.ACTIONS)}-D action, got {action.shape}")
    return {
        "model": str(model_dir), "device": device,
        "parameters": sum(p.numel() for p in policy.parameters()),
        "scene": f"{manifest.family} {manifest.variant} ({manifest.split})",
        "robot": "home pose", "prompt": prompt,
        "state_shape": list(batch["observation.state"].shape),
        "cameras": {key: [cam, list(batch[key].shape)] for key, cam in cameras.items()},
        "action_shape": list(action.shape), "action_names": list(Layout.ACTIONS),
        "action": [float(v) for v in action], "inference_seconds": round(seconds, 2),
        "note": "first action of one predicted chunk; not sent to the robot; not closed-loop "
                "control",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", type=Path, default=Path("models/smolvla_weeds"))
    parser.add_argument("--prompt", default="remove the shepherd weed")
    parser.add_argument("--family", default="midday_dry", choices=list(Manifest.FAMILIES))
    parser.add_argument("--variant", type=int, default=0)
    args = parser.parse_args()
    manifest = Manifest.variants(args.family, args.variant + 1)[args.variant]
    out_dir = Path("outputs/demo_logs") / time.strftime("%Y%m%d_%H%M%S")
    result = probe(args.model, args.prompt, manifest,
                   (Path("outputs/demo") / f"{args.family}_{args.variant}").resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smolvla_probe.json").write_text(json.dumps(result, indent=2))
    print("[SMOLVLA INFERENCE]")
    print(f"model={result['model']}")
    print(f"parameters={result['parameters']}")
    print(f"device={result['device']}")
    print(f"scene={result['scene']}, robot at its {result['robot']}")
    print(f'prompt="{result["prompt"]}"')
    print(f"state_dim={result['state_shape'][0]}")
    for key, (cam, shape) in result["cameras"].items():
        print(f"camera={key} <- {cam} {'x'.join(map(str, shape))}")
    print(f"action_dim={result['action_shape'][0]}")
    print(f"action_shape={tuple(result['action_shape'])}")
    print("action=[" + ", ".join(f"{v:.4f}" for v in result["action"]) + "]")
    for name, value in zip(result["action_names"], result["action"]):
        print(f"  {name:8s} {value:10.4f}")
    print(f"inference_seconds={result['inference_seconds']}")
    print(f"note={result['note']}")
    print(f"saved={out_dir / 'smolvla_probe.json'}")


if __name__ == "__main__":
    main()
