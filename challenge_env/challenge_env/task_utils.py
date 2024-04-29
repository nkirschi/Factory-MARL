import numpy as np
from typing import Optional
from dm_control import mjcf
import copy


class TaskManager:
    def __init__(
        self,
        physics: mjcf.Physics,
        mjcf_model: mjcf.RootElement,
        seed: Optional[int] = None,
        spawn_freq: int = 1 / 4,
    ):
        self.physics = physics
        self.mjcf_model = mjcf_model
        self.spawn_freq = spawn_freq

        self.rng = np.random.default_rng(seed)

        all_joints = self.mjcf_model.find_all("joint")
        objects = [j for j in all_joints if j.tag == "freejoint"]
        self.n_obj = len(objects)

        self.buckets = [
            g for g in self.mjcf_model.find_all("geom") if g.name == "target_area"
        ]

        self._all_objects = objects

        self._hide_x_pos = 4.0
        self._init_qpos = np.array([self._hide_x_pos, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])[
            None, :
        ].repeat(self.n_obj, axis=0)
        self._init_qpos[:, 1] += np.arange(self.n_obj) * 0.2
        self._init_qpos = self._init_qpos.flatten()
        self._spawn_pos = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    @property
    def terminate(self) -> bool:
        return self._failure_counter > 0

    @property
    def score(self, i: int) -> int:
        return self._scores[i]

    def _spawn_pose(self) -> np.ndarray:
        pose = self._spawn_pos.copy()  # randomize spawn posse
        pose[3:7] = self.rng.uniform(
            0, 1, (4,)
        )  # TODO: actually sample a valid rotation
        return pose

    def _spawn_object(self) -> None:
        if len(self._out_of_scene) == 0:
            return

        obj = self._out_of_scene.pop(0)
        self.physics.bind(obj).qpos[:] = self._spawn_pose()
        self._in_scene.append(obj)

    def get_valid_object_vectors(self) -> tuple[np.ndarray, np.ndarray, int]:
        num_valid = len(self._in_scene)
        qpos = self.physics.bind(self._in_scene).qpos.copy().reshape(-1, 7)
        qvel = self.physics.bind(self._in_scene).qvel.copy().reshape(-1, 6)

        # sort by x position
        sort_idx = np.argsort(qpos[:, 0])
        qpos = qpos[sort_idx]
        qvel = qvel[sort_idx]

        qpos_vec = np.zeros((self.n_obj, 7))
        qvel_vec = np.zeros((self.n_obj, 6))
        qpos_vec[:num_valid] = qpos
        qvel_vec[:num_valid] = qvel

        return qpos_vec, qvel_vec, num_valid

    def _check_states(self) -> None:
        if len(self._in_scene) == 0:
            return
        qpos = self.physics.bind(self._in_scene).qpos.copy().reshape(-1, 7)

        # Check if objects are out of scene (failures)
        x_bound = 1.2
        y_bound = 1.2
        z_bound = 0.9
        oox = np.abs(qpos[:, 0]) > x_bound
        ooy = np.abs(qpos[:, 1]) > y_bound
        ooz = qpos[:, 2] < z_bound
        out_of_scene = oox | ooy | ooz
        for obj_idx in sorted(np.where(out_of_scene)[0], reverse=True):
            obj = self._in_scene.pop(obj_idx)
            self._hide_object(obj)
            self._failure_counter += 1

        if len(self._in_scene) == 0:
            return

        # Check if objects are placed inside one of the buckets
        obj_pos = self.physics.bind(self._in_scene).qpos.copy().reshape(-1, 7)

        for i, b in enumerate(self.buckets):
            bucket_pos = self.physics.bind(b).xpos
            bucket_size = self.physics.bind(b).size
            in_x = np.abs(obj_pos[:, 0] - bucket_pos[0]) <= 0.6 * bucket_size[0]
            in_y = np.abs(obj_pos[:, 1] - bucket_pos[1]) <= 0.6 * bucket_size[1]
            in_z = obj_pos[:, 2] - bucket_pos[2] - bucket_size[2] / 2 <= 0.07
            in_bucket = in_x & in_y & in_z
            for obj_idx in sorted(np.where(in_bucket)[0], reverse=True):
                obj = self._in_scene.pop(obj_idx)
                self._hide_object(obj)
                self._scores[i] += 1

    def _hide_object(self, obj: mjcf.RootElement) -> None:
        self._out_of_scene.append(obj)
        self.physics.bind(obj).qpos[:] = np.array(
            [
                self._hide_x_pos + 1.0,
                self._out_of_scene_counter * 0.2,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.physics.bind(obj).qvel[:] = 0.0
        self._out_of_scene_counter += 1

    @property
    def spawn_steps(self) -> None:
        return int(1.0 / (0.1 * self.spawn_freq))

    def step(self) -> None:

        if self._step_counter == 0 or self._steps_since_spawn >= self.spawn_steps:
            self._spawn_object()
            self._steps_since_spawn = 0

        self._check_states()

        self._step_counter += 1
        self._steps_since_spawn += 1

    def reset(self) -> None:
        self.physics.bind(self._all_objects).qpos[:] = self._init_qpos.copy()
        self.physics.bind(self._all_objects).qvel[:] = 0.0
        self._out_of_scene = copy.copy(self._all_objects)
        self._in_scene = []

        self._step_counter = 0
        self._steps_since_spawn = 0
        self._failure_counter = 0
        self._out_of_scene_counter = 0
        self._scores = [0, 0]
