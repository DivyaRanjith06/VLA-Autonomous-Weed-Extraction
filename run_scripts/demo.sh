#!/bin/bash
# Type a prompt and watch the robot scan the bed, park at a matching weed and pick it.
# Uses the weed detector and the scripted expert; the detector is built
# here on first use. Example: ./run_scripts/demo.sh "remove the shepherd weed"
# Extra arguments go to the demo, e.g. --family low_light --variant 1 --headless

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

detector=models/detector.pt

if [[ ! -f "$detector" ]]; then
	.venv/bin/python -m weed_extraction.detector render outputs/detector/data
	.venv/bin/python -m weed_extraction.detector train outputs/detector/data "$detector"
fi

.venv/bin/python -m weed_extraction.demo "$@"
