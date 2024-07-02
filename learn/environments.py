import copy
import numpy as np
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces

from challenge_env.base_env import BaseEnv
from challenge_env.ik_policy import IKPolicy, PolicyState


class TaskEnv(BaseEnv):
    def __init__(
            self,
            render_mode: Optional[str] = None,
            seed: Optional[int] = None,
            width=1024,
            height=1024
    ):
        """
        This class defines a custom task environment, including custom observation and action spaces,
        and a custom reward function. It inherits from BaseEnv, which is a subclass of gym.Env.
        """

        super().__init__(render_mode=render_mode, seed=seed, width=width, height=height)

        self.ik_policy0 = IKPolicy(self, arm_id=0)
        self.ik_policy1 = IKPolicy(self, arm_id=1)

        # Example custom observation and action spaces:
        obs_dims = 6 * self.dof + 13 * self.max_num_objects
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dims),
            high=np.inf * np.ones(obs_dims),
            dtype=np.float32,
        )

        action_dims = self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _process_observation(self, state: np.ndarray) -> np.ndarray:
        """
        Your custom feature representation logic goes here.
        Input:
            state: np.ndarray containing the environment state
            as defined in challenge_specifications.STATE_DTYPE
        Output:
            obs: Your custom feature representation as expected by your agent
        """
        joint_states = np.concatenate([
            state["qpos_player0"].flatten(),
            state["qvel_player0"].flatten(),
            state["ctrl_player0"].flatten(),
            state["qpos_player1"].flatten(),
            state["qvel_player1"].flatten(),
            state["ctrl_player1"].flatten(),
        ])
        object_states = np.concatenate([
            state["object_poses"].flatten(),
            state["object_vels"].flatten()
        ])
        obs = np.concatenate([joint_states, object_states])
        return obs

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Your custom action processing logic may go here.
        Input:
            action: np.ndarray as output by your agent
        Output:
            ctrl: Scaled and clipped control signal as expected by the environment
        """
        # A common choice is to let the policy output actions in the range [-1, 1]
        # and scale them to the desired range here
        # action = np.clip(action, -1.0, 1.0)
        action = np.tanh(action)
        bounds = self.joint_limits.copy()  # joint_limits is a property of BaseEnv
        low, high = bounds.T
        ctrl = low + (action + 1.0) * 0.5 * (high - low)
        return ctrl

    def _compose_control(self, rl_action):
        """
        Compose the control action for the environment.

        :param rl_action: The action provided by the reinforcement learning algorithm.
        :return: The composed control action.
        """

        # Override this in a subclass
        action_arm0 = self.ik_policy0.act().clip(self.model.actuator_ctrlrange[1:9, 0],
                                                 self.model.actuator_ctrlrange[1:9, 1])
        self.ik_policy1.ignore(self.ik_policy0.target_object)
        action_arm1 = self.ik_policy1.act().clip(self.model.actuator_ctrlrange[9:17, 0],
                                                 self.model.actuator_ctrlrange[9:17, 1])
        self.ik_policy0.ignore(self.ik_policy1.target_object)
        return action_arm0, action_arm1

    def _get_reward(self, state, action, info) -> float:
        """
        Compute the reward for the current state of the environment.

        This method overrides the `_get_reward` method from the parent class `TaskEnv`.
        The reward is computed based on the base reward, the distance between the gripper and the cube,
        and the distance between the cube and the bucket.

        Parameters:
        state (np.ndarray): The current state of the environment.
        action (np.ndarray): The action taken in the current state.
        info (dict): Additional information about the current state.

        Returns:
        float: The computed reward.
        """

        reward = sum(info["scores"]) - sum(self.last_score)
        return reward

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_arm0, action_arm1 = self._compose_control(action)

        state, terminate, info = self.step_sim(
            action_arm0=action_arm0, action_arm1=action_arm1
        )

        # TODO terminate = ... dependent on info

        reward = self._get_reward(state, action, info)
        obs = self._process_observation(state)
        self.last_score = info["scores"].copy()
        return obs, reward, terminate, False, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.reset_sim(seed=seed)
        self.ik_policy0.reset()
        self.ik_policy1.reset()
        obs = self._process_observation(state)
        self.last_score = info["scores"]

        return obs, {}


class ProgressRewardEnv(TaskEnv):
    """
    This class extends the TaskEnv class and adds a distance penalty to the reward function.
    The distance penalty is calculated based on the distance between the gripper and the cube, and the distance between
    the cube and the bucket.
    Adds a base reward to the reward function to prevent large negative rewards that can cause early termination.
    """

    NUM_AGENTS = 2  # Number of arms in the environment, currently we assume it equals number of buckets

    def __init__(
            self,
            gripper_to_closest_cube_reward_factor,
            closest_cube_to_bucket_reward_factor,
            small_action_norm_reward_factor,
            base_reward=0.0,
            **kwargs
    ):
        """
        :param gripper_to_closest_cube_reward_factor: Weight for the gripper-to-closest-cube distance.
        :param closest_cube_to_bucket_reward_factor: Weight for the closest-cube-to-bucket distance.
        :param base_reward: A base reward to avoid large negative rewards.
        """
        super().__init__(**kwargs)
        self.gripper_to_closest_cube_reward_factor = gripper_to_closest_cube_reward_factor
        self.closest_cube_to_bucket_reward_factor = closest_cube_to_bucket_reward_factor
        self.small_action_norm_reward_factor = small_action_norm_reward_factor
        self.last_gripper_to_closest_cube_dist = [0] * self.NUM_AGENTS
        self.last_bucket_to_closest_cube_dist = [0] * self.NUM_AGENTS
        self.base_reward = base_reward

    def _compute_gripper_to_closest_cube_dist(self, ik_policy, last_dist_gripper_to_closest_cube):
        """
        Compute the distance between the given gripper and the closest cube on the scene that it is not ignoring.

        :param ik_policy: The inverse kinematics policy of the gripper to calculate the distance for.
        :param last_dist_gripper_to_closest_cube: The previous distance between the gripper and the closest cube.

        :return: The closest cube, its distance to the gripper, and the change in distance to the previous step.
        """
        gripper_pos = self.physics.named.data.site_xpos[ik_policy._gripper_site]

        candidates = copy.copy(self.task_manager._in_scene)
        if ik_policy.ignore_object is not None and ik_policy.ignore_object in candidates:
            candidates.remove(ik_policy.ignore_object)

        if len(candidates) == 0:
            return None, last_dist_gripper_to_closest_cube, 0

        cand_pos = self.physics.bind(candidates).qpos.reshape(-1, 7)[:, :3]
        dists = np.linalg.norm(cand_pos - gripper_pos, axis=1)
        min_index = np.argmin(dists)

        return candidates[min_index], dists[min_index], last_dist_gripper_to_closest_cube - dists[min_index]

    def _compute_bucket_to_closest_cube_dist(self, closest_cube, bucket_pos, last_dist_bucket_cube):
        """
        Compute the distance between the given cube and the closest bucket.

        :param closest_cube: The cube that is closest to the gripper.
        :param bucket_pos: The position of the bucket.
        :param last_dist_bucket_cube: The previous distance between the bucket and the cube.
        :return: The distance to the bucket and the change in distance to the previous step.
        """
        if closest_cube is None:
            return last_dist_bucket_cube, 0

        cube_pos = self.physics.bind(closest_cube).qpos[:3]
        dist_bucket_cube = np.linalg.norm(cube_pos - bucket_pos)

        return dist_bucket_cube, last_dist_bucket_cube - dist_bucket_cube

    def _get_reward(self, state, action, info) -> float:
        """
        see superclass
        """

        # Compute the closest cube for each agent and the distance to the cube
        gripper_cube_reward = 0.0
        policies = [self.ik_policy0, self.ik_policy1]
        closest_cubes = []
        for i, policy in enumerate(policies):
            closest_cube, dist_closest_cube, dist_closest_cube_change = self._compute_gripper_to_closest_cube_dist(
                policy, self.last_gripper_to_closest_cube_dist[i]
            )
            closest_cubes.append(closest_cube)
            self.last_gripper_to_closest_cube_dist[i] = dist_closest_cube
            gripper_cube_reward += dist_closest_cube_change

        # Compute the distance between the cube and the own bucket for each agent
        bucket_cube_reward = 0.0
        bucket_positions = [self.physics.bind(b).xpos for b in self.task_manager.buckets]
        for i, _ in enumerate(closest_cubes):
            dist_bucket, dist_bucket_change = self._compute_bucket_to_closest_cube_dist(
                closest_cubes[i], bucket_positions[i], self.last_bucket_to_closest_cube_dist[i]
            )
            self.last_bucket_to_closest_cube_dist[i] = dist_bucket
            bucket_cube_reward += dist_bucket_change

        # Compute a decreasing positive function of the action norm, ignoring the gripper dims
        non_gripper_actions = action[[i for i in range(self.action_space.shape[0]) if i % 8 != 7]]
        action_norm_reward = np.exp(-np.linalg.norm(non_gripper_actions))

        score_reward = super()._get_reward(state, action, info)
        progress_reward = self.base_reward \
                          + self.gripper_to_closest_cube_reward_factor * max(0.0, gripper_cube_reward) \
                          + self.closest_cube_to_bucket_reward_factor * max(0.0, bucket_cube_reward) \
                          + self.small_action_norm_reward_factor * action_norm_reward

        # If some agent scored a point, the score change is returned instead of the progress reward to avoid an
        # unwanted penalty due to the suddenly increased distance to the new closest cube
        return score_reward if score_reward > 0 else progress_reward


class SingleFullRLProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compose_control(self, rl_action):
        _, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = self._process_action(rl_action)
        action_arm1 = ik_action1
        return action_arm0, action_arm1


class SingleDeltaProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compose_control(self, rl_action):
        ik_action0, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = ik_action0
        if self.ik_policy0.state is not PolicyState.IDLE:
            action_arm0 += 0.25 * self._process_action(rl_action)
        action_arm1 = ik_action1
        return action_arm0, action_arm1


class DoubleDeltaProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_dims = 2 * self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, rl_action):
        ik_action0, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = ik_action0
        if self.ik_policy0.state is not PolicyState.IDLE:
            action_arm0 += 0.25 * self._process_action(rl_action[0:8])
        action_arm1 = ik_action1
        if self.ik_policy1.state is not PolicyState.IDLE:
            action_arm1 += 0.25 * self._process_action(rl_action[8:16])

        return action_arm0, action_arm1
