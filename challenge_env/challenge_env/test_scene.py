from dm_control import mjcf
from dm_control.utils import inverse_kinematics as ik
from dm_control import mujoco
import numpy as np
from scene import build_scene, set_initial_camera_params

if __name__ == "__main__":
    import time
    import mujoco

    mjcf_model = build_scene(num_objects=1, randomize_objects=False)
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

    object_pos = np.array([0, 0, 1.05 + 0.082])

    physics.named.data.qpos["cube0/"][:3] = object_pos.copy()

    physics.named.data.qpos["arm1/iiwa14/joint2"]

    # init ik pick up solution simulation
    physics.named.data.qpos["arm1/iiwa14/joint2"] = -0.796
    physics.named.data.qpos["arm1/iiwa14/joint4"] = 1.38
    physics.named.data.qpos["arm1/iiwa14/joint6"] = -0.859
    physics.named.data.qpos["arm1/iiwa14/joint7"] = -0.5
    physics.named.data.qpos["arm1/iiwa14/single_gripper/left_plate_slide_joint"] = 0.06
    physics.named.data.qpos["arm1/iiwa14/single_gripper/right_plate_slide_joint"] = 0.06

    # compute pick location
    grasp_plan = ik.qpos_from_site_pose(
        physics,
        "arm1/iiwa14/single_gripper/between_gripper_plates",
        target_pos=object_pos,
        target_quat=np.array([0, 1, 1, 0]),
        joint_names=None,
        tol=1e-14,
        rot_weight=1.0,
        regularization_threshold=0.1,
        regularization_strength=3e-2,
        max_update_norm=2.0,
        progress_thresh=20.0,
        max_steps=10000,
        inplace=False,
    )
    print(grasp_plan)
    # write first pick location
    physics.data.qpos = grasp_plan.qpos

    # init simulation
    physics.named.data.qpos["arm1/iiwa14/joint7"] = -0.5
    physics.named.data.qpos["arm1/iiwa14/single_gripper/left_plate_slide_joint"] = 0.06
    physics.named.data.qpos["arm1/iiwa14/single_gripper/right_plate_slide_joint"] = 0.06
    physics.named.data.ctrl["arm1/iiwa14/actuator1"] = physics.named.data.qpos[
        "arm1/iiwa14/joint1"
    ]
    physics.named.data.ctrl["arm1/iiwa14/actuator2"] = physics.named.data.qpos[
        "arm1/iiwa14/joint2"
    ]
    physics.named.data.ctrl["arm1/iiwa14/actuator3"] = physics.named.data.qpos[
        "arm1/iiwa14/joint3"
    ]
    physics.named.data.ctrl["arm1/iiwa14/actuator4"] = physics.named.data.qpos[
        "arm1/iiwa14/joint4"
    ]
    physics.named.data.ctrl["arm1/iiwa14/actuator5"] = physics.named.data.qpos[
        "arm1/iiwa14/joint5"
    ]
    physics.named.data.ctrl["arm1/iiwa14/actuator6"] = physics.named.data.qpos[
        "arm1/iiwa14/joint6"
    ]
    physics.named.data.ctrl["arm1/iiwa14/actuator7"] = -0.5
    physics.named.data.ctrl["arm1/iiwa14/single_gripper/gripper_linear_actuator"] = 0.0

    model = physics.model._model
    data = physics.data._data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_initial_camera_params(viewer)
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)

            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
