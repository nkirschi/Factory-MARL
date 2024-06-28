# Example for a custom task definition
import copy
import numpy as np
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces

from challenge_env.base_env import BaseEnv
from challenge_env.ik_policy import IKPolicy


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
        obs_dims = 2 * self.dof + 13 * self.max_num_objects
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dims),
            high=np.inf * np.ones(obs_dims),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-np.ones(self.dof), high=np.ones(self.dof), dtype=np.float32
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
        # Example:
        dof_measurements = np.concatenate(
            [
                state["qpos_player1"].flatten(),
                state["qvel_player1"].flatten(),
            ]
        )

        object_states = np.concatenate(
            [state["object_poses"].flatten(), state["object_vels"].flatten()]
        )
        obs = np.concatenate([dof_measurements, object_states])
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
        action = np.clip(action, -1.0, 1.0)
        bounds = self.joint_limits.copy()  # joint_limits is a property of BaseEnv
        low, high = bounds.T
        ctrl = low + (action + 1.0) * 0.5 * (high - low)
        return ctrl

    def _compose_control(self, rl_action):
        # Override this in a subclass
        action_arm0 = self.ik_policy0.act()
        self.ik_policy1.ignore(self.ik_policy0.target_object)
        action_arm1 = self.ik_policy1.act()
        self.ik_policy0.ignore(self.ik_policy1.target_object)
        return action_arm0, action_arm1

    def _get_reward(self, state, action, info) -> float:
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


class SingleDeltaEnv(TaskEnv):
    def __init__(
            self,
            score_weight,
            norm_penalty_weight,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.score_weight = score_weight
        self.norm_penalty_weight = norm_penalty_weight

    def _compose_control(self, rl_action):
        ik_action0, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = ik_action0 + 0.5 * self._process_action(rl_action)
        # TODO: debug output of act() and force-clip if necessary
        action_arm1 = ik_action1
        return action_arm0, action_arm1

    def _get_reward(self, state, rl_action, info) -> float:
        # Your custom reward function goes here
        score_reward = super()._get_reward(state, rl_action, info)
        norm_reward = np.exp(-np.linalg.norm(self._process_action(rl_action)))
        # TODO: take norm of [-1,1] action but ignore gripper dimension
        return self.score_weight * score_reward + self.norm_penalty_weight * norm_reward


class SingleFullRLEnv(TaskEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compose_control(self, rl_action):
        _, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = self._process_action(rl_action)
        action_arm1 = ik_action1
        return action_arm0, action_arm1

    def _get_reward(self, state, action, info) -> float:
        # Your custom reward function goes here
        return super()._get_reward(state, action, info) + 0.1 * np.exp(-np.linalg.norm(self._process_action(action)))


class TaskEnvWithDistancePenalty(TaskEnv):
    """
    This class extends the TaskEnv class and adds a distance penalty to the reward function.
    The distance penalty is calculated based on the distance between the gripper and the cube, and the distance between
    the cube and the bucket.
    Adds a base reward to the reward function to prevent large negative rewards that can cause early termination.
    """

    NUM_AGENTS = 2  # Number of arms in the environment, currently we assume it equals number of buckets

    def __init__(
            self,
            distance_gripper_reward_factor,
            distance_bucket_reward_factor,
            base_reward=0.0,
            render_mode: Optional[str] = None,
            seed: Optional[int] = None,
            width=1024,
            height=1024
    ):
        """
        Initialize the environment with the given parameters.

        :param distance_gripper_reward_factor: The reward factor for the distance between the gripper and the cube.
        :param distance_bucket_reward_factor: The reward factor for the distance between the cube and the bucket.
        :param base_reward: The base reward for the environment, it gets added to the reward to avoid large negative
        rewards.
        :param render_mode: The mode to render the environment.
        :param seed: The seed for the random number generator.
        :param width: The width of the environment.
        :param height: The height of the environment.
        """
        super().__init__(render_mode=render_mode, seed=seed, width=width, height=height)
        self.distance_bucket_reward_factor = distance_bucket_reward_factor
        self.distance_gripper_reward_factor = distance_gripper_reward_factor
        self.old_min_dist_grabber_cube = [0] * self.NUM_AGENTS
        self.old_min_dist_cube_bucket = [0] * self.NUM_AGENTS
        self.base_reward = base_reward

    def _compose_control(self, rl_action):
        """
        Compose the control action for the environment.

        :param rl_action: The action provided by the reinforcement learning algorithm.
        :return: The composed control action.
        """
        ik_action0, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = ik_action0 + 0.5 * self._process_action(rl_action)
        action_arm1 = ik_action1
        return action_arm0, action_arm1

    def _compute_cube_grabber_distance(self, ik_policy, old_min_dist_grabber_cube):
        """
        Compute the distance between the cube and the grabber.

        :param ik_policy: The inverse kinematics policy.
        :param old_min_dist_grabber_cube: The old minimum distance between the grabber and the cube.

        :return: The closest cube, the minimum distance, and the change in distance.
        """
        position = self.physics.named.data.site_xpos[ik_policy._gripper_site]

        candidates = copy.copy(self.task_manager._in_scene)
        if ik_policy.ignore_object is not None:
            if ik_policy.ignore_object in candidates:
                candidates.remove(ik_policy.ignore_object)

        if len(candidates) == 0:
            return None, old_min_dist_grabber_cube, 0

        pos = self.physics.bind(candidates).qpos.reshape(-1, 7)[:, :3]
        dists = np.linalg.norm(
            pos - position,
            axis=1,
        )

        dist_to_cube_change = old_min_dist_grabber_cube - np.min(dists)

        closest_object = candidates[np.argmin(dists)]
        return closest_object, np.min(dists), dist_to_cube_change

    def _compute_bucket_cube_distance(self, closest_cube, bucket_pos, old_min_bucket_dist):
        """
        Compute the distance between the cube and the bucket.

        :param closest_cube: The cube that is closest to the grabber.
        :param bucket_pos: The position of the bucket.
        :param old_min_bucket_dist: The old minimum distance between the bucket and the cube.
        :return: The distance to the bucket and the change in distance.
        """
        if closest_cube is None:
            return old_min_bucket_dist, 0
        cube_position = self.physics.bind(closest_cube).qpos[:3]
        dist_cube_bucket = np.linalg.norm(cube_position - bucket_pos)
        dist_bucket_change = old_min_bucket_dist - dist_cube_bucket

        return dist_cube_bucket, dist_bucket_change

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
        reward = super()._get_reward(state, action, info)
        # Check if the agent scored a point, if so, the distance to the cube and bucket calculated to get a difference
        # in the next round, but the score change is then return.
        # Otherwise, the much larger distance to the now next closest cube is adding so much penalty, that the reward
        # will be low or negative.
        scored = reward > 0

        new_reward = reward + self.base_reward

        policies = [self.ik_policy0, self.ik_policy1]

        # Compute the closest cube for each agent and the distance to the cube
        closest_cubes = []
        for i, policy in enumerate(policies):
            closest_cube, dist_closest_cube, dist_closest_cube_change \
                = self._compute_cube_grabber_distance(policy, self.old_min_dist_grabber_cube[i])
            self.old_min_dist_grabber_cube[i] = dist_closest_cube
            closest_cubes.append(closest_cube)
            self.old_min_dist_grabber_cube[i] = dist_closest_cube
            new_reward = new_reward + self.distance_gripper_reward_factor * dist_closest_cube_change

        # Reads the bucket positions
        bucket_positions = [None] * self.NUM_AGENTS
        for (i, b) in enumerate(self.task_manager.buckets):
            bucket_pos = self.physics.bind(b).xpos
            bucket_positions[i] = bucket_pos

        # Compute the distance between the cube and the bucket
        for i, closest_cube in enumerate(closest_cubes):
            bucket_pos = bucket_positions[i]
            dist_to_bucket, dist_to_bucket_change = self._compute_bucket_cube_distance(closest_cube, bucket_pos,
                                                                                       self.old_min_dist_cube_bucket[i])
            self.old_min_dist_cube_bucket[i] = dist_to_bucket
            new_reward = new_reward + self.distance_bucket_reward_factor * dist_to_bucket_change

        return reward if scored else new_reward
