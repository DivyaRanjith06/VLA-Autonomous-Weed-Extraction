"""The SmolVLA input boundary: the dataset as recorded, converted only where the policy
needs it.

SmolVLA passes every camera through its SigLIP vision encoder, which takes three-channel
images in [0, 1]. The wrist depth is stored as the camera measured it -- one channel of
metres -- so a processor step at the head of SmolVLA's own preprocessor turns it into
that form. The dataset keeps the raw depth; training and inference both go through the
same preprocessor, and the step's settings are saved with it, so a trained policy
converts depth exactly as it was trained on.

To fine-tune pretrained SmolVLA with LeRobot's standard training script, first write a
model directory for the packed dataset, then point the script at it:

    python -m weed_extraction.policy <packed_dataset_split> <model_dir>
    python -m lerobot.scripts.lerobot_train --policy.path=<model_dir> \
        --env.discover_packages_path=weed_extraction \
        --dataset.repo_id=<repo_id> --dataset.root=<packed_dataset_split> ...

The discover flag imports this package so LeRobot knows the depth step. It must not
carry the `policy.` prefix: LeRobot re-reads those as policy settings and rejects it.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from weed_extraction.pack import Layout  # also puts the vendored LeRobot on the path
from lerobot.configs import PipelineFeatureType, PolicyFeature  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.processor.pipeline import (ObservationProcessorStep,  # noqa: E402
                                        ProcessorStepRegistry)
from lerobot.utils.feature_utils import dataset_to_policy_features  # noqa: E402


@dataclass
@ProcessorStepRegistry.register(name="weed_extraction_depth_to_rgb")
class DepthToRGBProcessorStep(ObservationProcessorStep):
    """Metric depth, one channel, as a three-channel image in [0, 1]: depth over `max_m`,
    clipped, the same in every channel. Nearer is darker."""

    key: str = Layout.CAMERAS["d435i_depth"]
    # The unit the depth arrives in: LeRobot's dataset loader returns millimetres unless
    # asked otherwise.
    unit: str = "mm"
    # The wrist camera works within arm's reach: in the smoke data 90% of its depth lies
    # within 0.73 m and all of it but open sky within 1.5 m. 2 m keeps that detail;
    # anything farther reads as far.
    max_m: float = 2.0

    def observation(self, observation):
        depth = observation[self.key]
        metres = depth / 1000.0 if self.unit == "mm" else depth
        image = (metres / self.max_m).clamp(0.0, 1.0)
        channels = image.dim() - 3  # (C, H, W), with any batch dims in front
        observation[self.key] = image.repeat_interleave(3, dim=channels)
        return observation

    def get_config(self):
        """Saved with the preprocessor, so a reloaded one converts exactly alike."""
        return {"key": self.key, "unit": self.unit, "max_m": self.max_m}

    def transform_features(self, features):
        observations = dict(features[PipelineFeatureType.OBSERVATION])
        _, height, width = observations[self.key].shape
        observations[self.key] = PolicyFeature(type=FeatureType.VISUAL,
                                               shape=(3, height, width))
        return {**features, PipelineFeatureType.OBSERVATION: observations}


def policy_features(dataset_features):
    """SmolVLA's input and output features for a packed dataset's features: the depth as
    the step delivers it, and nothing outside `observation.*` and `action`."""
    features = dataset_to_policy_features(dataset_features)
    transformed = DepthToRGBProcessorStep().transform_features({
        PipelineFeatureType.OBSERVATION: {k: f for k, f in features.items()
                                          if f.type is not FeatureType.ACTION},
        PipelineFeatureType.ACTION: {k: f for k, f in features.items()
                                     if f.type is FeatureType.ACTION}})
    return transformed[PipelineFeatureType.OBSERVATION], transformed[PipelineFeatureType.ACTION]


def make_processors(config, dataset_stats):
    """SmolVLA's preprocessor with the depth step at its head, and its postprocessor."""
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset_stats)
    preprocessor.steps.insert(0, DepthToRGBProcessorStep())
    return preprocessor, postprocessor


def write_model_dir(dataset_root, out_dir, checkpoint="lerobot/smolvla_base"):
    """A SmolVLA model directory for `lerobot_train.py --policy.path`, fitted to the packed
    dataset at `dataset_root`: the checkpoint's weights, symlinked; its config with the
    dataset's policy features; and SmolVLA's processors, depth step first, carrying the
    dataset's statistics. The cameras keep their own names. Returns `out_dir`.
    """
    dataset_root, out_dir = Path(dataset_root), Path(out_dir)
    labels = json.loads((dataset_root / "meta" / "weed_extraction" / "labels.json").read_text())
    meta = LeRobotDatasetMetadata(labels["repo_id"], root=dataset_root)
    weights = Path(hf_hub_download(checkpoint, "model.safetensors")).resolve()
    config_file = Path(hf_hub_download(checkpoint, "config.json"))
    config = PreTrainedConfig.from_pretrained(config_file.parent)
    config.input_features, config.output_features = policy_features(meta.features)
    # The checkpoint holds the whole VLM, so its separate weights need not be fetched.
    config.load_vlm_weights = False
    processors = make_processors(config, meta.stats)
    # Loading fell back to this machine's device if it lacks the checkpoint's. The config
    # keeps the checkpoint's, so a GPU machine trains on its GPU; the processors' device
    # step is replaced by the policy's device whenever lerobot_train loads them.
    config.device = json.loads(config_file.read_text())["device"]
    out_dir.mkdir(parents=True)
    config.save_pretrained(out_dir)
    (out_dir / "model.safetensors").symlink_to(weights)
    for processor in processors:
        processor.save_pretrained(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=write_model_dir.__doc__.split("\n\n")[0])
    parser.add_argument("dataset_root", type=Path, help="one packed split, e.g. .../train")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--checkpoint", default="lerobot/smolvla_base")
    args = parser.parse_args()
    print(write_model_dir(args.dataset_root, args.out_dir, args.checkpoint))


if __name__ == "__main__":
    main()
