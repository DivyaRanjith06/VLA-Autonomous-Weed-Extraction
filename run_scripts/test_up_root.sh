#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# First run on a fresh clone: build the virtualenv.
[ -x .venv/bin/weed-launch ] || command -v weed-launch >/dev/null || ./run_scripts/setup.sh || exit 1

# Natively the launcher lives in the virtualenv; in the container it is already on PATH.
LAUNCH=.venv/bin/weed-launch
[ -x "$LAUNCH" ] || LAUNCH=weed-launch

exec "$LAUNCH" uproot "$@"
