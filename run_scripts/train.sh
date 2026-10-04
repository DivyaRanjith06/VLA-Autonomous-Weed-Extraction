#!/bin/bash
# Fine-tune SmolVLA into the pick model: bed and wrist cameras, colour only, vision frozen.
# Needs run_scripts/setup.sh --smolvla and the pick-only dataset:
#   .venv/bin/python -m weed_extraction.dataset outputs/dataset_pick --parked --chunk-size 8
# Extra arguments go to LeRobot's training script, e.g. ./run_scripts/train.sh --steps=40

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

dataset=outputs/dataset_pick/lerobot/train
model=outputs/smolvla_pick

if [[ ! -d "$model" ]]; then
	.venv/bin/python -m weed_extraction.policy "$dataset" "$model" --cameras bed wrist
fi

# Batch 16 uses about 5 GB of GPU memory and takes about 1.3 s a step on an RTX 4060, so
# 10,000 steps take about 3.6 hours: about 3 passes over the 102-episode pick dataset.
# The learning-rate schedule shrinks itself to fit a run shorter than its 30,000 steps.
# pyav reads the videos: torchcodec needs FFmpeg libraries this setup does not install.
.venv/bin/python -m lerobot.scripts.lerobot_train \
	--policy.path="$model" \
	--env.discover_packages_path=weed_extraction \
	--dataset.repo_id=weed_extraction/dataset_pick_train \
	--dataset.root="$dataset" \
	--dataset.video_backend=pyav \
	--policy.push_to_hub=false \
	--wandb.enable=false \
	--batch_size=16 \
	--num_workers=4 \
	--steps=10000 \
	--save_freq=2500 \
	--output_dir=outputs/train/smolvla_pick \
	"$@"
