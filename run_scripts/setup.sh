#!/bin/bash
# One-time native setup: the Panda meshes and a virtualenv with the package in it.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

git submodule update --init workspace/hardware/mujoco_menagerie
python3.12 -m venv .venv
.venv/bin/pip install -e "workspace/planning/weed_extraction[dev]"
