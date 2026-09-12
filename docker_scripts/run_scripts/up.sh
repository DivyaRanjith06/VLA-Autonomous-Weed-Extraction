#!/bin/bash

# Starts the container and opens the simulation in it.
# Usage: ./up.sh [view|capture] [--key=value ...]

./devel.sh || exit 1

docker exec -it vla_weed weed-launch "${@:-view}"
