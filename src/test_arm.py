from pathlib import Path
import mujoco
import mujoco.viewer

scene_path = (
    Path(__file__).resolve().parent.parent
    / "external"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(scene_path))
data = mujoco.MjData(model)

# Panda home position
home = [
    0.0,
    0.0,
    0.0,
    -1.57079,
    0.0,
    1.57079,
    -0.7853,
    255.0,
]

# Slightly different arm configuration
target = [
    0.3,
    -0.4,
    0.2,
    -1.8,
    0.0,
    1.6,
    -0.6,
    255.0,
]

with mujoco.viewer.launch_passive(model, data) as viewer:

    # Start at home
    data.ctrl[:] = home

    for _ in range(1000):
        mujoco.mj_step(model, data)
        viewer.sync()

    # Move to target
    data.ctrl[:] = target

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
