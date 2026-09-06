from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

scene_path = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model, height=480, width=640)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        renderer.update_scene(data, camera="rgb_camera")
        image = renderer.render()
        image = np.flipud(image)

        cv2.imshow("Agent View", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        viewer.sync()

cv2.destroyAllWindows()