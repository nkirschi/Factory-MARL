from dm_control import mjcf
import mujoco.viewer
import numpy as np
import os

DIR = os.path.dirname(os.path.abspath(__file__)) + "/assets"


class PickableObject:
    def __init__(self, name, size=0.04, color=None):
        if color is None:
            color = [0.0, 0.3, 0.7, 1.0]
        mass = 1000 * size**3
        self.mjcf_model = mjcf.RootElement(model=name)
        self.body = self.mjcf_model.worldbody.add(
            "geom",
            name="cube",
            type="box",
            size=[size] * 3,
            mass=mass,
            rgba=color,
            friction=[1.0, 0.01, 0.01],
        )


class Table:
    def __init__(self):
        self.mjcf_model = mjcf.RootElement(model="table")
        self.body = self.mjcf_model.worldbody.add(
            "geom",
            name="table",
            type="box",
            pos=[0.0, 0.0, 0.5],
            size=[1.2, 1, 0.5],
            solimp=[0.98, 0.9999, 0.001],
            solref=[0.002, 1],
            priority=1,
        )


class Arm:
    def __init__(self, pos, name, flip=False):
        self.mjcf_model = mjcf.RootElement(model=name)
        euler = [0, 0, 0]
        if flip:
            euler = [0, 0, np.pi]

        self.site = self.mjcf_model.worldbody.add(
            "site",
            name=f"player_site",
            pos=pos,
            euler=euler,
        )
        arm = mjcf.from_path(DIR + "/kuka_iiwa_14/iiwa14.xml")
        gripper = mjcf.from_path(DIR + "/gripper.xml")
        self.tcp = arm.find("site", "attachment_site")
        self.tcp.attach(gripper)
        self.site.attach(arm)


class BucketFence:
    def __init__(self, size, width=0.05, height=0.05):
        color = [0.2, 0.2, 0.2, 1.0]
        self.mjcf_model = mjcf.RootElement(model="bucket_fence")
        self.body = self.mjcf_model.worldbody.add(
            "geom",
            name="fence",
            type="box",
            size=[width, size, height],
            pos=[size - width, 0.0, 0.0],
            rgba=color,
            solimp=[0.98, 0.9999, 0.001],
            solref=[0.002, 1],
            priority=1,
        )


class Bucket:
    def __init__(self, size=0.3):
        self.mjcf_model = mjcf.RootElement(model="bucket")
        self.body = self.mjcf_model.worldbody.add(
            "body",
            name="bucket",
        )
        self.body.add(
            "geom",
            type="box",
            name="target_area",
            size=[size - 0.01, size - 0.01, 0.02],
            pos=[0.0, 0.0, -0.04],
            rgba=[1.0, 1.0, 1.0, 1.0],
            solimp=[0.98, 0.9999, 0.001],
            solref=[0.002, 1],
            priority=1,
        )

        for f in range(4):
            frame = self.body.add(
                "site",
                name=f"fence_site{f}",
                euler=[0, 0, f * 1.57],
            )
            frame.attach(BucketFence(size).mjcf_model)


def build_scene(num_objects=10, randomize_objects=True, seed=None):

    mjcf_model = mjcf.from_path(DIR + "/scene.xml")

    # table
    table = Table()
    mjcf_model.attach(table.mjcf_model)

    # conveyor belt
    conveyor_belt = mjcf.from_path(DIR + "/conveyor_belt.xml")
    mjcf_model.attach(conveyor_belt)

    rng = np.random.default_rng(seed)

    # objects
    for i in range(num_objects):
        if randomize_objects:
            size = rng.uniform(0.03, 0.05)
            color = rng.uniform(0, 1, (4,))
            color[3] = 1.0
        else:
            size = 0.04
            color = [0.0, 0.3, 0.7, 1.0]
        cube = PickableObject(f"cube{i}", size=size, color=color)
        obj_frame = mjcf_model.attach(cube.mjcf_model)
        obj_frame.add("freejoint")

    # buckets
    bucket_dist_x = 0.9
    bucket_dist_y = 0.7
    bucket0 = Bucket()
    bucket0.body.set_attributes(pos=[bucket_dist_x, bucket_dist_y, 1.05])
    mjcf_model.attach(bucket0.mjcf_model)

    bucket1 = Bucket()
    bucket1.body.set_attributes(pos=[-bucket_dist_x, bucket_dist_y, 1.05])
    mjcf_model.attach(bucket1.mjcf_model)

    # arms
    dist_x = 0.6
    arm1 = Arm([dist_x, 0, 1.0], name="arm0")
    mjcf_model.attach(arm1.mjcf_model)

    arm2 = Arm([-dist_x, 0, 1.0], name="arm1", flip=True)
    mjcf_model.attach(arm2.mjcf_model)

    return mjcf_model


def set_initial_camera_params(viewer):
    # viewer.lock()
    viewer.cam.lookat = np.array([-0.30914206, -0.14805237, 1.53675732])
    viewer.cam.distance = 3.6720494497198732
    viewer.cam.azimuth = 66.95742187499998
    viewer.cam.elevation = -28.84335937500002


if __name__ == "__main__":
    import time
    import mujoco

    mjcf_model = build_scene()
    print(mjcf_model.to_xml_string())
    # physics = mjcf.Physics.from_xml_string(mjcf_model.to_xml_string(), assets=mjcf_model.get_assets())
    physics = mjcf.Physics.from_mjcf_model(mjcf_model)

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
