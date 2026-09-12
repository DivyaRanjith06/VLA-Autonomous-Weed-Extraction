#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Natively the launcher lives in the virtualenv; in the container it is already on PATH.
LAUNCH=.venv/bin/weed-launch
[ -x "$LAUNCH" ] || LAUNCH=weed-launch

exec "$LAUNCH" "${@:-view}"
