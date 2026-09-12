import os
import sys

import gymnasium as gym

# macOS: offscreen rendering must use CGL; GLFW contexts fail off the main thread
# (the viewer runs user code off the main thread under mjpython). Set before mujoco loads.
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")

import mujoco  # noqa: E402  (must follow MUJOCO_GL)

# Left to itself MuJoCo appends warnings to a MUJOCO_LOG.TXT in whatever directory
# you happened to run from. Send them to stderr instead, where they are visible.
mujoco.set_mju_user_warning(lambda message: print(f"mujoco: {message}", file=sys.stderr))

gym.register(id="WeedExtraction-v0", entry_point="weed_extraction.env:WeedExtractionEnv")
