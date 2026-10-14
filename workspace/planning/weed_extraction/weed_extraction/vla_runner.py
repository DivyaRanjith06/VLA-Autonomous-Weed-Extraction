"""SmolVLA in the loop: camera images, robot state and a prompt in, one 9-D action out per
control tick, applied to the MuJoCo scene.

    runner = VLARunner(Path("models/smolvla_weeds"))
    runner.load(manifest, scene_dir)
    runner.start("remove the shepherd weed", matching=["shepherd_1", ...])
    while (step := runner.step()) is not None: ...

Every tick builds the observation the policy was trained on (pack.Layout: the 28-D state,
the bed and wrist colour images at the training resolution) and asks the policy for an
action. SmolVLA keeps a queue of 50 actions: it runs inference only when the queue is
empty, and otherwise hands out the next queued action. The action is applied unchanged
through env.step; nothing is interpolated, repeated or filled in while inference runs.

The environment does not take a grasp by itself: the hold on a weed is an explicit weld
(env.Grasp) that the acting controller engages. The bridge here engages it as the
scripted expert does -- jaws commanded shut and both fingertips on a weed -- with a small
tolerance on the shut command, since the policy's gripper output is continuous, and
releases it when the jaws are commanded open. Every weed is checked: what the jaws close
on is decided by the physics, not by the prompt.
"""

import os

# Before LeRobot and transformers load: the checkpoint and tokenizer come from the local
# cache, never the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from weed_extraction.demo import ActionLog  # noqa: E402
from weed_extraction.pack import Layout  # noqa: E402  (also puts LeRobot on the path)
from weed_extraction.recorder import bin_interior, ee_pose, gripper_width, in_bin  # noqa: E402
from weed_extraction.scene import build_env  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
import weed_extraction.policy  # noqa: E402,F401  (registers the depth step the processors may name)


class VLAActionLog(ActionLog):
    """The demo's action log, for actions the policy produced."""

    SOURCE = "smolvla"


class VLARunner:
    """Owns one scene and the policy; steps them together. Not thread-safe: one thread
    calls load, start and step."""

    CONTROL_HZ = 25  # the rate dataset v2 was recorded at

    def __init__(self, model_dir, device=None, grip_close=10.0, grip_open=245.0):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = PreTrainedConfig.from_pretrained(model_dir)
        config.device = self.device
        self.policy = SmolVLAPolicy.from_pretrained(model_dir, config=config).eval()
        override = {"device_processor": {"device": self.device}}
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=str(model_dir), preprocessor_overrides=override,
            postprocessor_overrides=override)
        self.cameras = {key: cam for cam, key in Layout.CAMERAS.items()
                        if key in config.input_features}
        if any(cam.endswith("_depth") for cam in self.cameras.values()):
            raise ValueError("this runner feeds colour cameras only")
        _, self.size, _ = config.input_features[next(iter(self.cameras))].shape
        self.grip_close, self.grip_open = grip_close, grip_open
        self.env = self.log = None

    def load(self, manifest, scene_dir):
        """Build `manifest`'s scene with the training cameras, resolution and rate."""
        self.close()
        self.manifest = manifest
        self.env = build_env(manifest, scene_dir, cameras=tuple(self.cameras.values()),
                             width=self.size, height=self.size, control_hz=self.CONTROL_HZ,
                             max_steps=10**9)
        self.obs, _ = self.env.reset()
        model = self.env.model
        self.qpos = [model.joint(j).qposadr[0] for j in Layout.JOINTS]
        self.qvel = [model.joint(j).dofadr[0] for j in Layout.JOINTS]
        self.prompt = self.weed = None
        return self.obs

    def start(self, prompt, matching):
        """Reset the scene to its start and the policy's action queue, for `prompt`.
        `matching` names the weeds that fit the prompt; one in the bin ends the run."""
        self.obs, _ = self.env.reset()
        self.policy.reset()
        self.prompt, self.matching = prompt, list(matching)
        self.steps, self.inferences = 0, 0
        return self.obs

    def needs_inference(self):
        """True when the next step runs the policy rather than taking a queued action."""
        return self.policy._check_get_actions_condition()

    def step(self):
        """Ask the policy for one action and apply it. Returns the applied action."""
        if self.needs_inference():
            self.inferences += 1
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(self.batch())))
        action = action.cpu().numpy().reshape(-1).astype(np.float64)
        if action.shape != (len(Layout.ACTIONS),):
            raise RuntimeError(f"the policy gave a {action.shape} action, not 9 values")
        self.bridge_grasp(action[7])
        self.obs, _, _, _, info = self.env.step(action)
        self.steps += 1
        if self.log is not None:
            self.log.record(self, "smolvla", None)
        return self.env.data.ctrl.copy()  # as applied: the environment clips to ctrlrange

    def batch(self):
        model, data = self.env.model, self.env.data
        position, quat = ee_pose(model, data)
        state = np.concatenate([data.qpos[self.qpos], data.qvel[self.qvel], position, quat,
                                [gripper_width(model, data)]]).astype(np.float32)
        batch = {"observation.state": torch.from_numpy(state), "task": self.prompt}
        for key, cam in self.cameras.items():
            image = np.ascontiguousarray(self.obs[cam])
            batch[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        return batch

    def bridge_grasp(self, command):
        """Engage the existing grasp weld when the jaws are commanded shut on a weed, and
        release it when they are commanded open."""
        grasp, data = self.env.grasp, self.env.data
        if command <= self.grip_close and grasp.held is None:
            weed = next((w for w in grasp.welds if grasp.closed(data, w)), None)
            if weed is not None:
                grasp.take(data, weed)
        elif command >= self.grip_open:
            grasp.let_go(data)

    def done(self):
        """A weed that fits the prompt lies in the bin."""
        model, data = self.env.model, self.env.data
        lo, hi = bin_interior(model, data)
        return any(in_bin(data.xpos[model.body(w).id], lo, hi) for w in self.matching)

    @property
    def time(self):
        return float(self.env.data.time)

    def frames(self):
        """The current wrist and bed camera images, as rendered for the policy."""
        return {key.rsplit(".", 1)[-1]: self.obs[cam] for key, cam in self.cameras.items()}

    def close(self):
        """Free the scene's renderer. The policy stays loaded for the next scene."""
        if self.env is not None:
            self.env.close()
            self.env = None
