from pathlib import Path
import mujoco
import numpy as np
from PIL import Image

scene_path = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

renderer.update_scene(data, camera="rgb_camera")
image = renderer.render()
image = np.flipud(image)

Image.fromarray(image).save("camera.png")

print("Saved camera.png")