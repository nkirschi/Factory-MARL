from dm_control.utils import inverse_kinematics as ik
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
import enum
from absl import logging


# scipy quaternions are scalar-last, mujoco is scalar-first
def quat_scipy2mujoco(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def quat_mujoco2scipy(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])


class PolicyState(enum.Enum):
    IDLE = 0
    GO_TO_GRASP = 1
    GRASP_APPROACH = 2
    GRASP_CLOSE = 3
    POST_GRASP = 4
    GO_TO_RELEASE = 5
    RELEASE = 6


class IKPolicy:
    def __init__(self, env, arm_id: int = 0, verbosity=logging.ERROR, bucket_idx=None):
        self.physics = env.physics
        self.env = env
        self.arm_id = arm_id
        self._gripper_joint = env.actuated_joints[arm_id][-1]
        self._ik_joints = env.actuated_joints[arm_id][
            :-1
        ]  # don't include gripper in IK
        self._ik_joint_ids = [
            self.physics.model.name2id(joint, "joint") for joint in self._ik_joints
        ]
        self._gripper_site = (
            f"arm{self.arm_id}/iiwa14/single_gripper/between_gripper_plates"
        )

        self._base_site = self.physics.named.data.site_xpos[
            f"arm{self.arm_id}/player_site"
        ]
        if bucket_idx is None and arm_id>1:
            raise ValueError("bucket_idx must be specified for arm_id > 1")
        if bucket_idx is None:
            bucket_idx = arm_id
        self._bucket_pos = self.physics.bind(self.env.task_manager.buckets[bucket_idx]).xpos

        # parameters defining behavior
        self.workspace_radius = 1.0
        self.grasp_radius = 0.8
        self.pt_compensation = (
            env.pt_time * env.dt * 15.0
        )  # simple heuristic for compensating for object movement
        self.pre_grasp_height = 0.15
        self.post_grasp_height = 0.18
        self.target_threshold = 0.05
        self.release_threshold = 0.1
        self.grasp_offset = 0.04
        self.release_wait = int(0.5 / env.dt)
        self.grasp_wait = int(1.0 / env.dt)
        self.move_steps = int(1.0 / env.dt)
        self.timeout_steps = int(3.0 / env.dt)

        self.pre_release_pos = np.array([self._bucket_pos[0], self._bucket_pos[1], 1.3])

        self.default_pose = np.array([-0.5, -0.5, 0.0, 1.0, 0.0, -1.6, 0.0, 2.0])
        self.default_gripper_quat = np.array([0, 0, 1, 0])

        self.cube_sym = R.create_group("O")

        self.last_ctrl = self.default_pose
        self.state = PolicyState.IDLE
        self.target_object = None
        self.ignore_object = None
        self._move_start_pos = None
        self.state_counter = 0

        logging.set_verbosity(verbosity)

    def ignore(self, object):
        self.ignore_object = object

    def lin_interp(self, from_pos, to_pos):
        t = self.state_counter / self.move_steps
        return from_pos + (to_pos - from_pos) * t

    def select_target_object(self):

        candidates = copy.copy(self.env.task_manager._in_scene)

        if self.ignore_object is not None:
            if self.ignore_object in candidates:
                candidates.remove(self.ignore_object)

        # check if the current target object is still feasible
        if self.target_object is not None:
            obj_pos = self.physics.bind(self.target_object).qpos[:3]
            dist = np.linalg.norm(obj_pos - self._base_site)
            if dist < self.workspace_radius and self.target_object in candidates:
                return self.target_object

        if len(candidates) == 0:
            return None

        pos = self.physics.bind(candidates).qpos.reshape(-1, 7)[:, :3]
        speculative_pos = pos.copy() 
        speculative_pos[:, 1] -= 0.2
        dists = np.linalg.norm(
            speculative_pos - self._base_site[None, :],
            axis=1,
        )
        closest_object = candidates[np.argmin(dists)]
        if dists[np.argmin(dists)] < self.grasp_radius:
            return closest_object
        else:
            return None

    def reset(self):
        self.idle_ctrl()

    def idle_ctrl(self):
        self.set_state(PolicyState.IDLE)
        self.target_object = None
        self.last_ctrl = self.default_pose
        return self.last_ctrl

    def set_state(self, state):
        self.state_counter = 0
        self.state = state

    @property
    def target_object_pose(self):
        return self.physics.bind(self.target_object).qpos.copy()

    @property
    def target_object_vel(self):
        return self.physics.bind(self.target_object).qvel[:3].copy()

    def act(self):

        self.target_object = self.select_target_object()

        if self.target_object is None:
            return self.idle_ctrl()

        # get current object state
        object_pose = self.target_object_pose
        object_pos = object_pose[:3]
        object_rot = object_pose[3:]
        object_vel = self.target_object_vel

        # find gripper rotation ik target
        # This is the rotation that is aligned with the cube symmetry group
        # and closest to the default gripper rotation
        obj_rot_sp = R.from_quat(quat_mujoco2scipy(object_rot))
        obj_sym_rots = obj_rot_sp * self.cube_sym
        default_gripper_sp = R.from_quat(quat_mujoco2scipy(self.default_gripper_quat))
        sym_dists = R.magnitude(default_gripper_sp * obj_sym_rots.inv())
        nearest_cube_sym = obj_sym_rots[np.argmin(sym_dists)]
        target_quat = quat_scipy2mujoco(nearest_cube_sym.as_quat())

        gripper_pos = self.physics.named.data.site_xpos[self._gripper_site]

        pre_grasp_pos = object_pos.copy()
        pre_grasp_pos[2] += self.pre_grasp_height

        grasp_pos = object_pos.copy()
        grasp_pos[2] += self.grasp_offset

        near_object = np.linalg.norm(gripper_pos - object_pos) < self.grasp_offset

        if self.state == PolicyState.IDLE:
            idle_dist = np.linalg.norm(
                self.physics.named.data.qpos[self._ik_joints] - self.default_pose[:-1]
            )
            if idle_dist < 0.1:
                self.set_state(PolicyState.GO_TO_GRASP)
        elif self.state == PolicyState.GO_TO_GRASP:
            dist = np.linalg.norm(gripper_pos - pre_grasp_pos)
            if dist < self.target_threshold:
                self.set_state(PolicyState.GRASP_APPROACH)
        elif self.state == PolicyState.GRASP_APPROACH:
            if near_object:
                self.set_state(PolicyState.GRASP_CLOSE)
        elif self.state == PolicyState.GRASP_CLOSE:
            if near_object and self.state_counter > self.grasp_wait:
                self._move_start_pos = gripper_pos.copy()
                self.set_state(PolicyState.POST_GRASP)
            elif not near_object:
                self.set_state(PolicyState.IDLE)
        elif self.state == PolicyState.POST_GRASP:
            if not near_object:
                self.set_state(PolicyState.IDLE)
            else:
                dist = np.linalg.norm(
                    gripper_pos[2] - (self._move_start_pos[2] + self.post_grasp_height)
                )
                if dist < self.target_threshold:
                    self._move_start_pos = gripper_pos.copy()
                    self.set_state(PolicyState.GO_TO_RELEASE)
        elif self.state == PolicyState.GO_TO_RELEASE:
            if not near_object:
                self.set_state(PolicyState.IDLE)
            else:
                dist = np.linalg.norm(gripper_pos - self.pre_release_pos)
                if dist < self.release_threshold:
                    self.set_state(PolicyState.RELEASE)
        elif self.state == PolicyState.RELEASE:
            if self.state_counter > self.release_wait:
                self.set_state(PolicyState.IDLE)

        close_gripper = False
        compensate_movement = False

        if self.state == PolicyState.IDLE:
            return self.idle_ctrl()
        elif self.state == PolicyState.GO_TO_GRASP:
            target_pos = pre_grasp_pos
            compensate_movement = True
        elif self.state == PolicyState.GRASP_APPROACH:
            target_pos = grasp_pos
            compensate_movement = True
        elif self.state == PolicyState.GRASP_CLOSE:
            close_gripper = True
            target_pos = grasp_pos
            compensate_movement = True
        elif self.state == PolicyState.POST_GRASP:
            close_gripper = True
            end_pos = self._move_start_pos.copy()
            end_pos[2] += self.post_grasp_height
            target_pos = self.lin_interp(self._move_start_pos, end_pos)

            target_quat = self.default_gripper_quat.copy()
        elif self.state == PolicyState.GO_TO_RELEASE:
            close_gripper = True
            target_pos = self.lin_interp(
                self._move_start_pos, self.pre_release_pos.copy()
            )
            target_quat = self.default_gripper_quat.copy()
        elif self.state == PolicyState.RELEASE:
            close_gripper = False
            target_pos = self.pre_release_pos.copy()
            target_quat = self.default_gripper_quat.copy()

        self.state_counter += 1

        if self.state_counter > self.timeout_steps:
            return self.idle_ctrl()

        # compensate for object movement
        if compensate_movement:
            target_pos[:2] += object_vel[:2] * self.pt_compensation

        # find ik solution
        ik_sol = ik.qpos_from_site_pose(
            self.physics,
            self._gripper_site,
            target_pos=target_pos.copy(),
            target_quat=target_quat.copy(),
            joint_names=self._ik_joints,
            max_steps=10,
        )
        # print(ik_sol.steps)
        if not ik_sol.success:
            return self.last_ctrl

        # TODO: check if copying physics object is slow
        physics2 = self.physics.copy(share_model=True)
        physics2.data.qpos[:] = ik_sol.qpos
        q_targets = physics2.named.data.qpos[self._ik_joints]
        ctrl = np.zeros(8)
        ctrl[: len(q_targets)] = q_targets

        if close_gripper:
            ctrl[-1] = 0.0
        else:
            ctrl[-1] = 2.0

        self.last_ctrl = ctrl.copy()
        return ctrl


if __name__ == "__main__":
    from challenge_env.base_env import BaseEnv

    env = BaseEnv(render_mode="human")
    env.reset_sim()

    policy = IKPolicy(env, arm_id=0)
    policy1 = IKPolicy(env, arm_id=1)

    for t in range(10000):
        ctrl0 = policy.act()
        policy1.ignore(policy.target_object)
        ctrl1 = policy1.act()
        policy.ignore(policy1.target_object)
        # ctrl1 = None
        state, terminate, info = env.step_sim(ctrl0, ctrl1)

        if terminate:
            env.reset_sim()

    env.close()
