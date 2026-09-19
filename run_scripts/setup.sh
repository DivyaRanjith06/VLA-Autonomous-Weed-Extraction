#!/bin/bash
# One-time native setup: the Panda meshes and a virtualenv with the package in it.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

git submodule update --init workspace/hardware/mujoco_menagerie
python_bin="${PYTHON_BIN:-}"
if [[ -z "$python_bin" ]]; then
	for candidate in python3.14 python3.13 python3.12; do
		if command -v "$candidate" >/dev/null 2>&1; then
			python_bin="$candidate"
			break
		fi
	done
fi

if [[ -z "$python_bin" ]]; then
	cat >&2 <<'EOF'
Python 3.12 or newer is required, but no supported interpreter was found.

On Ubuntu 22.04, install Python 3.12 from the Deadsnakes repository, then rerun:
	sudo apt install software-properties-common
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt update
	sudo apt install python3.12 python3.12-venv

Or use the Docker setup under docker_scripts/run_scripts/.
EOF
	exit 1
fi

python_version="$($python_bin -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python_major="${python_version%%.*}"
python_minor="${python_version##*.}"
if ((python_major < 3 || (python_major == 3 && python_minor < 12))); then
	echo "Python 3.12 or newer is required; $python_bin provides Python $python_version." >&2
	exit 1
fi

echo "Creating .venv with $python_bin (Python $python_version)..."
"$python_bin" -m venv .venv
.venv/bin/pip install -e "workspace/planning/weed_extraction[dev]"
