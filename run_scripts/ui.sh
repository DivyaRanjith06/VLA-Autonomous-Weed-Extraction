#!/bin/bash
# Browser UI: pick a held-out scene, type a prompt, watch the demo's state machine act in the
# MuJoCo viewer. Needs models/detector.pt (Git LFS).
# Example: ./run_scripts/ui.sh            (opens http://127.0.0.1:8765/ in the browser)
# Extra arguments go to the UI, e.g. --seconds 90 --port 8000 --no-browser --detector <file>

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ROS's Python 3.10 packages on PYTHONPATH do not belong in this 3.12 environment.
exec env -u PYTHONPATH .venv/bin/python -m weed_extraction.ui "$@"
