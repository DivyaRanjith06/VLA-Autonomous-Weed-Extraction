ARG ROS_DISTRO=jazzy

# Ubuntu 24.04 with the ROS 2 core, built for amd64 and arm64.
# https://hub.docker.com/_/ros
# 24.04 is what supplies Python 3.12. The ROS base is here so that the ROS 2 packages
# this workspace is laid out for can be dropped in later without changing the image.
FROM ros:$ROS_DISTRO-ros-base

# Stop apt asking questions mid-build.
ARG DEBIAN_FRONTEND=noninteractive

# Clearing the lists first avoids the hash-mismatch failures a stale cache causes.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update -yqqq

# Somewhere for GLFW to put its runtime files, or it warns on every launch.
RUN mkdir /tmp/runtime-root && chmod 0700 /tmp/runtime-root
ENV XDG_RUNTIME_DIR=/tmp/runtime-root

# General tools.
RUN apt-get install --no-install-recommends -yqqq \
    git \
    ca-certificates \
    vim \
    ffmpeg \
    python3-venv

# Rendering. Mesa gives EGL for offscreen work on any machine, with or without a GPU;
# with the NVIDIA container toolkit installed, EGL picks the NVIDIA driver instead.
# The X libraries are what the interactive viewer needs to draw on the host's display.
RUN apt-get install --no-install-recommends -yqqq \
    libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri \
    libglfw3 libx11-6 libxcursor1 libxrandr2 libxinerama1 libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Where the repo lives inside the container.
ARG WORKSPACE=/root/VLA-Weed-Extraction
ENV WORKSPACE=$WORKSPACE

# Ubuntu 24.04 marks its Python as externally managed, so pip needs a virtualenv.
# System site packages stay visible so rclpy is importable from the same interpreter.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The whole workspace, so the image can run on its own. devel.sh mounts the repo over
# it, and then the mounted source is what runs.
COPY workspace $WORKSPACE/workspace

# build.sh --build-arg WITH_SMOLVLA=true adds LeRobot and the dataset/SmolVLA libraries
# (the smolvla extra); LeRobot comes from its submodule, with --no-deps for the same reason
# as in run_scripts/setup.sh.
ARG WITH_SMOLVLA=false
RUN if [ "$WITH_SMOLVLA" = true ]; then \
        pip install --no-cache-dir -e "$WORKSPACE/workspace/planning/weed_extraction[dev,smolvla]" \
        && pip install --no-cache-dir --no-deps -e "$WORKSPACE/workspace/planning/lerobot"; \
    else \
        pip install --no-cache-dir -e "$WORKSPACE/workspace/planning/weed_extraction[dev]"; \
    fi

# MUJOCO_GL is deliberately unset: MuJoCo picks GLFW when a display is forwarded and
# EGL when there is none, so one image serves both.
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=$ROS_DISTRO

RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc

WORKDIR $WORKSPACE
CMD ["bash"]
