#!/bin/bash

docker build --progress="${VLA_BUILD_PROGRESS:-auto}" -t vla_weed:latest "$@" ../..
