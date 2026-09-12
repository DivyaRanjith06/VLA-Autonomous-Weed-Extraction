#!/bin/bash

./stop.sh

# The viewer draws on the host's X display, so the socket and DISPLAY are passed in.
xhost +local:root 2>/dev/null

# Same as base.sh, but the repo is mounted over the copy baked into the image, so your
# edits take effect with no rebuild.
docker run -t -d --net=host \
--name vla_weed \
--shm-size=2g \
-v $PWD/../..:/root/VLA-Weed-Extraction \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--env DISPLAY="$DISPLAY" \
--env ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
vla_weed:latest
