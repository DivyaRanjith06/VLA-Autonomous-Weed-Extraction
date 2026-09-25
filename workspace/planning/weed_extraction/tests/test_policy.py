import json
import os
import subprocess
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the SmolVLM tokenizer comes from the cache

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="needs the smolvla extra: run_scripts/setup.sh --smolvla")

import torch  # noqa: E402
from huggingface_hub import hf_hub_download

from weed_extraction.pack import LeRobotDataset, Layout
from weed_extraction.policy import (DepthToRGBProcessorStep, make_processors, policy_features,
                                    write_model_dir)
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402

CAMERAS = ["d435i_color", "d435i_depth", "rgb_camera"]
DEPTH = Layout.CAMERAS["d435i_depth"]


def convert(depth, **settings):
    step = DepthToRGBProcessorStep(**settings)
    return step.observation({step.key: depth})[step.key]


# The depth step

def test_depth_becomes_three_identical_channels_in_the_unit_interval():
    depth_mm = torch.tensor([[[[300.0, 1000.0, 2000.0, 10000.0]]]])  # (B, 1, H, W)
    image = convert(depth_mm)
    assert image.shape == (1, 3, 1, 4)
    assert torch.equal(image[:, 0], image[:, 1]) and torch.equal(image[:, 1], image[:, 2])
    assert image[0, 0, 0].tolist() == pytest.approx([0.15, 0.5, 1.0, 1.0])


def test_depth_in_metres_converts_alike():
    assert torch.equal(convert(torch.tensor([[[0.3, 1.0, 5.0]]]), unit="m"),
                       convert(torch.tensor([[[300.0, 1000.0, 5000.0]]])))


def test_an_unbatched_frame_converts_like_a_batched_one():
    depth = torch.rand(1, 36, 64) * 3000
    assert torch.equal(convert(depth.clone()), convert(depth[None].clone())[0])


def test_the_raw_depth_is_left_as_it_was():
    depth = torch.rand(2, 1, 36, 64) * 3000
    before = depth.clone()
    convert(depth)
    assert torch.equal(depth, before)


# Features

@pytest.fixture(scope="module")
def dataset_features():
    """What pack writes to info.json, in its (H, W, C) layout."""
    return {k: {**f, "shape": list(f["shape"])} for k, f in
            Layout((144, 256), CAMERAS).features.items()}


def test_the_policy_sees_28_state_values_three_rgb_images_and_acts_in_nine(dataset_features):
    inputs, outputs = policy_features(dataset_features)
    assert {k: tuple(f.shape) for k, f in inputs.items()} == {
        "observation.state": (28,),
        "observation.images.wrist": (3, 144, 256),
        "observation.images.wrist_depth": (3, 144, 256),
        "observation.images.overview": (3, 144, 256)}
    assert {k: tuple(f.shape) for k, f in outputs.items()} == {"action": (9,)}
    assert dataset_features["observation.state"]["dtype"] == "float32"
    assert dataset_features["action"]["dtype"] == "float32"


def test_labels_and_the_root_wrench_are_never_policy_inputs(dataset_features):
    inputs, _ = policy_features(dataset_features)
    assert "label.root_wrench" in dataset_features
    assert not any(k.startswith("label.") or "wrench" in k for k in inputs)
    assert not any("wrench" in name for name in Layout.STATE)


# The preprocessor

@pytest.fixture(scope="module")
def processors(dataset_features):
    inputs, outputs = policy_features(dataset_features)
    config = SmolVLAConfig(input_features=inputs, output_features=outputs, device="cpu")
    return make_processors(config, dataset_stats=None)


def test_the_depth_step_heads_the_smolvla_preprocessor(processors):
    preprocessor, _ = processors
    assert isinstance(preprocessor.steps[0], DepthToRGBProcessorStep)
    assert sum(isinstance(s, DepthToRGBProcessorStep) for s in preprocessor.steps) == 1


def test_a_saved_preprocessor_converts_depth_exactly_as_the_trained_one(processors, tmp_path):
    """Inference loads the preprocessor saved at training: the depth step and its
    settings must come back with it."""
    preprocessor, _ = processors
    preprocessor.save_pretrained(tmp_path)
    reloaded = type(preprocessor).from_pretrained(tmp_path,
                                                  config_filename=f"{preprocessor.name}.json")
    step = reloaded.steps[0]
    assert isinstance(step, DepthToRGBProcessorStep)
    assert (step.key, step.unit, step.max_m) == (DEPTH, "mm", 2.0)
    # The settings are saved, not merely the defaults found again on reload.
    saved = json.loads((tmp_path / f"{preprocessor.name}.json").read_text())["steps"][0]
    assert saved["config"] == {"key": DEPTH, "unit": "mm", "max_m": 2.0}
    depth = torch.rand(2, 1, 36, 64) * 3000
    assert torch.equal(step.observation({DEPTH: depth})[DEPTH],
                       preprocessor.steps[0].observation({DEPTH: depth})[DEPTH])


# The model directory for LeRobot's standard training script

@pytest.fixture(scope="module")
def checkpoint():
    try:
        return hf_hub_download("lerobot/smolvla_base", "model.safetensors")
    except Exception:  # not cached, and these tests do not download 0.9 GB
        pytest.skip("lerobot/smolvla_base is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """A small dataset in exactly the packer's layout: one 60-frame episode."""
    root = tmp_path_factory.mktemp("packed") / "train"
    repo_id = "weed_extraction/test_train"
    layout = Layout((36, 64), CAMERAS)
    dataset = LeRobotDataset.create(repo_id, fps=50, features=layout.features, root=root,
                                    robot_type="mobile_panda")
    rng = np.random.default_rng(0)
    for _ in range(60):
        dataset.add_frame({
            "observation.state": rng.normal(size=28).astype(np.float32),
            "action": rng.normal(size=9).astype(np.float32),
            "label.root_wrench": rng.normal(size=2).astype(np.float32),
            **{label: np.array([1], np.int64) for label in Layout.LABELS},
            "observation.images.wrist": rng.integers(0, 256, (36, 64, 3), np.uint8),
            "observation.images.overview": rng.integers(0, 256, (36, 64, 3), np.uint8),
            DEPTH: rng.uniform(0.1, 1.5, (36, 64, 1)).astype(np.float32),
            "task": "remove weed 1"})
    dataset.save_episode()
    dataset.finalize()
    (root / "meta" / "weed_extraction").mkdir()
    (root / "meta" / "weed_extraction" / "labels.json").write_text(json.dumps({"repo_id": repo_id}))
    return root, repo_id


@pytest.fixture(scope="module")
def model_dir(checkpoint, packed, tmp_path_factory):
    return write_model_dir(packed[0], tmp_path_factory.mktemp("model") / "smolvla")


def test_the_model_directory_links_the_checkpoint_weights(checkpoint, model_dir):
    weights = model_dir / "model.safetensors"
    assert weights.is_symlink() and weights.resolve() == Path(checkpoint).resolve()


def test_its_config_is_the_checkpoint_fitted_to_the_dataset(model_dir, packed):
    config = json.loads((model_dir / "config.json").read_text())
    pretrained = json.loads(Path(hf_hub_download("lerobot/smolvla_base", "config.json")).read_text())
    assert {k: v["shape"] for k, v in config["input_features"].items()} == {
        "observation.state": [28],
        "observation.images.wrist": [3, 36, 64],
        "observation.images.wrist_depth": [3, 36, 64],
        "observation.images.overview": [3, 36, 64]}
    assert {k: v["shape"] for k, v in config["output_features"].items()} == {"action": [9]}
    assert config["load_vlm_weights"] is False
    unchanged = {k: v for k, v in pretrained.items()
                 if k not in ("input_features", "output_features", "load_vlm_weights")}
    assert {k: config[k] for k in unchanged} == unchanged


def test_its_preprocessor_converts_depth_first_and_renames_nothing(model_dir):
    steps = json.loads((model_dir / "policy_preprocessor.json").read_text())["steps"]
    assert steps[0]["registry_name"] == "weed_extraction_depth_to_rgb"
    assert steps[0]["config"] == {"key": DEPTH, "unit": "mm", "max_m": 2.0}
    rename = next(s for s in steps if s["registry_name"] == "rename_observations_processor")
    assert rename["config"]["rename_map"] == {}
    normalizer = next(s for s in steps if s["registry_name"] == "normalizer_processor")
    assert normalizer["config"]["features"]["observation.state"]["shape"] == [28]
    assert (model_dir / "policy_postprocessor.json").exists()


@pytest.mark.slow
def test_lerobot_train_consumes_the_model_directory(model_dir, packed, tmp_path):
    """The standard training script, one step, exactly as documented in policy.py."""
    root, repo_id = packed
    lerobot_src = Path(__file__).resolve().parents[2] / "lerobot" / "src"
    run = subprocess.run(
        [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--policy.path={model_dir}",
         "--env.discover_packages_path=weed_extraction", "--policy.device=cpu",
         "--policy.push_to_hub=false", f"--dataset.repo_id={repo_id}", f"--dataset.root={root}",
         "--steps=1", "--batch_size=1", "--num_workers=0", "--save_checkpoint=false",
         "--wandb.enable=false", "--log_freq=1", f"--output_dir={tmp_path / 'train'}"],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(lerobot_src)},
        capture_output=True, text=True, timeout=1200)
    log = run.stdout + run.stderr
    assert run.returncode == 0, log[-3000:]
    assert "step:1" in log and "End of training" in log
