from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from dm_control import mjcf
import time

from challenge_env.rendering import MujocoRenderer
from challenge_env.scene import build_scene
from challenge_env.task_utils import TaskManager

RESOLUTION = 480


class BaseEnv(gym.Env):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = RESOLUTION,
        height: int = RESOLUTION,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
        default_camera_config: Optional[Dict[str, Union[float, int]]] = None,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
        initial_conveyor_speed: float = 0.1,  # m/s
        conveyor_acceleration: float = 0.001,  # m/s^2
        pt_time: float = 0.2,  # s
        force_contact_threshold: float = 200.0,  # threshold for contact forces with arms, above which the episode is terminated
        max_num_objects: int = 10,
        control_frequency: float = 10,  # Hz
        spawn_freq: float = 1 / 5,
        spawn_freq_increase: float = 1.001,
        num_arms: int = 2,
    ):

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": 10,
        }

        self.width = width
        self.height = height

        self.max_num_objects = max_num_objects
        self.num_arms = num_arms
        self.physics = self._initialize_simulation(seed=seed)

        self.task_manager = TaskManager(
            self.physics, self.mjcf_model, seed=seed, spawn_freq=spawn_freq
        )

        self.model = self.physics.model
        self.data = self.physics.data

        self.renderer = MujocoRenderer(
            self.model._model,
            self.data._data,
            default_camera_config,
            self.width,
            self.height,
            max_geom,
            camera_id,
            camera_name,
            visual_options,
        )

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.init_state = self.physics.get_state()

        assert self.metadata["render_modes"] == [
            "human",
            "rgb_array",
        ], self.metadata["render_modes"]
        self.render_frame_skip = int((1.0 / 20.0) / self.model.opt.timestep)
        self.render_dt = self.render_frame_skip * self.model.opt.timestep

        self.dof = dof = 8

        arm_states = []
        for i in range(num_arms):
            arm_states.append((f"qpos_player{i}", "f4", (dof,)))  # joint positions
            arm_states.append((f"qvel_player{i}", "f4", (dof,)))  # joint velocities
            arm_states.append((f"ctrl_player{i}", "f4", (dof,)))  # joint control commands

        self.state_dtype = np.dtype(
            [
                *arm_states,  # joint states
                # object_poses and object_vels are constant-sized arrays that contain the state of the graspable objects in the scene
                # Only the first 'num_obj' entries contain valid data, the remaining entries are filled with zeros
                (
                    "object_poses",
                    "f4",
                    (self.max_num_objects, 7),
                ),  # zero-padded object positions and quaternion rotation (x, y, z, q0, q1, q2, q3)
                (
                    "object_vels",
                    "f4",
                    (self.max_num_objects, 6),
                ),  # zero-padded object velocities (linear and angular)
                ("num_obj", "i4"),  # current number of objects in the scene
            ]
        )

        # indexing
        self.actuated_joints = []
        for i in range(num_arms):
            joint = [f"arm{i}/iiwa14/joint{j}" for j in range(1, 8)] + [
                f"arm{i}/iiwa14/single_gripper/left_plate_slide_joint"
            ]
            self.actuated_joints.append(joint)

        self.joint_limits = self.physics.model.actuator_ctrlrange[1:9]

        self.arm_geoms = []
        for i in range(num_arms):
            for j in range(0, 61):
                self.arm_geoms.append(f"arm{i}/iiwa14//unnamed_geom_{j}")
            for j in range(9):
                self.arm_geoms.append(f"arm{i}/iiwa14/single_gripper//unnamed_geom_{j}")

        self.arm_geom_ids = np.array(
            [self.model.name2id(name, "geom") for name in self.arm_geoms]
        )

        # task parameters
        self.control_frequency = control_frequency  # Hz
        self.frame_skip = int((1 / self.control_frequency) / self.model.opt.timestep)
        self.pt_time = pt_time
        self.initial_conveyor_speed = initial_conveyor_speed
        self.conveyor_acceleration = conveyor_acceleration
        self.spawn_freq_increase = spawn_freq_increase
        self.init_spawn_freq = spawn_freq

        self.force_contact_threshold = force_contact_threshold

        self.render_mode = render_mode
        self.camera_name = camera_name
        self.camera_id = camera_id

        self.play_time = 0.0

    def _get_state(self) -> np.ndarray:

        arms_states = []
        # player 1
        for i in range(self.num_arms):
            qpos = self.physics.named.data.qpos[self.actuated_joints[i]]
            qvel = self.physics.named.data.qvel[self.actuated_joints[i]]
            ctrl_idx = i * 8
            ctrl = self.ctrl_target[ 1 + ctrl_idx : ctrl_idx + 9]
            arms_states.append(qpos)
            arms_states.append(qvel)
            arms_states.append(ctrl)

        # objects
        object_pose, object_vel, num_obj = self.task_manager.get_valid_object_vectors()

        state = np.array(
            (
                *arms_states,
                object_pose,
                object_vel,
                num_obj,
            ),
            dtype=self.state_dtype,
        )

        return state

    def reset_sim(
        self,
        *,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        super().reset(seed=seed)

        self.physics.reset()
        self.task_manager.reset()
        self.conveyor_speed = np.array([self.initial_conveyor_speed])
        self.task_manager.spawn_freq = self.init_spawn_freq
        self.ctrl_target = np.zeros(self.physics.model.nu)
        self.physics.after_reset()
        self.play_time = 0.0
        self.tick = time.time()

        state = self._get_state()

        if self.render_mode == "human":
            self.render()

        return state, {"scores": self.scores}

    @property
    def dt(self) -> float:
        return self.model.opt.timestep * self.frame_skip

    def _step_sim_unscaled(self, ctrl) -> Tuple[np.ndarray, bool]:
        force_terminate = False

        for t in range(self.frame_skip):
            # calculate ctrl_target by low-pass filtering the action
            self.ctrl_target += (ctrl - self.ctrl_target) * (
                self.model.opt.timestep / (self.model.opt.timestep + self.pt_time)
            )

            self.ctrl_target[0] = (
                -self.conveyor_speed
            )  # don't pt control the conveyor speed

            self.physics.set_control(self.ctrl_target.copy())
            self.physics.step()

            if self.render_mode == "human" and t % self.render_frame_skip == 0:
                self.render()
                time.sleep(max(0, self.render_dt - (time.time() - self.tick)))
                self.tick = time.time()
        # end episode if contact forces with arms are too high
        force_terminate = False
        if self.physics.data.ncon > 0:
            c_geom = self.physics.data.contact.geom

            arm_contacts = np.isin(c_geom, self.arm_geom_ids).any(axis=1)
            if arm_contacts.sum() > 0:
                forces = [
                    self.physics.data.contact_force(i)
                    for i in arm_contacts.nonzero()[0]
                ]
                if np.max(np.absolute(forces)) > self.force_contact_threshold:
                    force_terminate = True

        return force_terminate

    def step_sim(self, **kwargs) -> Tuple[np.ndarray, bool]:

        action_arm_n = []

        for i in range(self.num_arms):
            if f"action_arm{i}" in kwargs:
                action_arm_n.append(kwargs.get(f"action_arm{i}"))
            else:
                action_arm_n.append(np.zeros(self.dof))
        
        action_arm_n = np.concatenate(action_arm_n)       
            
        assert len(action_arm_n) == (self.num_arms) * self.dof , f"Control dimension mismatch. Expected {self.num_arms}*{self.dof}, found {len(action_arm_n)}"


        ctrl = np.concatenate([self.conveyor_speed, action_arm_n])

        # clip action
        ctrl = np.clip(
            ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

        force_terminate = self._step_sim_unscaled(ctrl)

        self.task_manager.step()
        self.play_time += self.dt

        self.conveyor_speed += self.conveyor_acceleration * self.dt
        self.task_manager.spawn_freq *= self.spawn_freq_increase

        terminate = self.task_manager.terminate or force_terminate

        info = {
            "scores": self.scores,
            "play_time": self.play_time,
            "conveyor_speed": self.conveyor_speed,
            "out_of_reach": self.task_manager.terminate,
            "force_terminate": force_terminate,
        }

        return self._get_state(), terminate, info

    @property
    def scores(self) -> np.ndarray:
        return self.task_manager._scores

    def render(self) -> np.ndarray:
        return self.renderer.render(self.render_mode, scores=self.scores)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    def _initialize_simulation(
        self,
        seed=None,
    ) -> mjcf.Physics:

        mjcf_model = build_scene(num_objects=self.max_num_objects, seed=seed, num_arms=self.num_arms)
        self.mjcf_model = mjcf_model
        physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        physics.model._model.vis.global_.offwidth = self.width
        physics.model._model.vis.global_.offheight = self.height

        return physics
