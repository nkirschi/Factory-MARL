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
            **kwargs
    ):
        """
        This class defines a custom task environment, including custom observation and action spaces,
        and a custom reward function. It inherits from BaseEnv, which is a subclass of gym.Env.
        """

        super().__init__(render_mode=render_mode, seed=seed, **kwargs)

        self.ik_policies = [IKPolicy(self, arm_id=i, bucket_idx=i % 2) for i in range(self.num_arms)]

        obs_dims = self.num_arms * 3 * self.dof + 13 * self.max_num_objects
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dims),
            high=np.inf * np.ones(obs_dims),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-np.empty(0),
            high=np.empty(0),
            dtype=np.float32
        )

        self.ep_score_history = []

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
            np.concatenate([
                state[f"qpos_player{i}"].flatten(),
                state[f"qvel_player{i}"].flatten(),
                state[f"ctrl_player{i}"].flatten(),
            ])
            for i in range(self.num_arms)
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
        arm_actions = []
        for i in range(self.num_arms):
            lo, hi = self.model.actuator_ctrlrange[1:9, 0], self.model.actuator_ctrlrange[1:9, 1]
            arm_actions.append(self.ik_policies[i].act().clip(lo, hi))
            for j in range(self.num_arms):
                if j != i:
                    self.ik_policies[j].ignore(self.ik_policies[i].target_object, i)
        return arm_actions

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
        arm_actions = self._compose_control(action)

        state, terminate, info = self.step_sim(
            **{f"action_arm{i}": arm_actions[i] for i in range(self.num_arms)}
        )

        reward = self._get_reward(state, action, info)
        obs = self._process_observation(state)
        self.last_score = info["scores"].copy()
        if terminate:
            self.ep_score_history.append(self.last_score)
        return obs, reward, terminate, False, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.reset_sim(seed=seed)
        for ik_policy in self.ik_policies:
            ik_policy.reset()
        obs = self._process_observation(state)
        self.last_score = info["scores"].copy()

        return obs, {}


class ProgressRewardEnv(TaskEnv):
    """
    This class extends the TaskEnv class and adds a distance penalty to the reward function.
    The distance penalty is calculated based on the distance between the gripper and the cube, and the distance between
    the cube and the bucket.
    Adds a base reward to the reward function to prevent large negative rewards that can cause early termination.
    """

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
        self.last_gripper_to_closest_cube_dist = [0] * self.num_arms
        self.last_bucket_to_closest_cube_dist = [0] * self.num_arms
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
        candidates = list(filter(lambda x: x not in ik_policy.ignore_objects.values(), candidates))

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
        policies = self.ik_policies
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
                closest_cubes[i], bucket_positions[i % 2], self.last_bucket_to_closest_cube_dist[i]
            )
            self.last_bucket_to_closest_cube_dist[i] = dist_bucket
            bucket_cube_reward += dist_bucket_change

        # Compute a decreasing positive function of the action norm, ignoring the gripper dims
        non_gripper_actions = action[[i for i in range(self.action_space.shape[0]) if i % 8 != 7]]
        action_norm_reward = np.exp(-np.linalg.norm(non_gripper_actions))

        score_reward = super()._get_reward(state, action, info)
        progress_reward = self.base_reward \
                          + self.gripper_to_closest_cube_reward_factor * gripper_cube_reward \
                          + self.closest_cube_to_bucket_reward_factor * bucket_cube_reward \
                          + self.small_action_norm_reward_factor * action_norm_reward

        # If some agent scored a point, the score change is returned instead of the progress reward to avoid an
        # unwanted penalty due to the suddenly increased distance to the new closest cube
        return score_reward if score_reward > 0 else progress_reward


class SingleFullRLProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_dims = self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, rl_action):
        ik_actions = super()._compose_control(None)
        return [self._process_action(rl_action)] + ik_actions[1:]


class SingleDeltaProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_dims = self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, rl_action):
        ik_actions = super()._compose_control(None)
        if self.ik_policies[0].state is not PolicyState.IDLE:
            ik_actions[0] += 0.5 * self._process_action(rl_action)
        return ik_actions


class AllFullRLProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_dims = self.num_arms * self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, rl_action):
        return [self._process_action(rl_action[self.dof*i:self.dof*(i+1)]) for i in range(self.num_arms)]


class AllDeltaProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_dims = self.num_arms * self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, rl_action):
        ik_actions = super()._compose_control(None)
        for i in range(self.num_arms):
            if self.ik_policies[i].state is not PolicyState.IDLE:
                ik_actions[i] += 0.5 * self._process_action(rl_action[self.dof*i:self.dof*(i+1)])
        return ik_actions


class IKToggleEnv(TaskEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.action_space = spaces.MultiDiscrete(self.num_arms * [2])

        obs_dims = self.num_arms * 3 * self.dof + 13 * self.max_num_objects + self.dof * self.num_arms  # add proposed IK actions
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dims),
            high=np.inf * np.ones(obs_dims),
            dtype=np.float32,
        )

    def _process_observation(self, state: np.ndarray) -> np.ndarray:
        obs = super()._process_observation(state)
        self.ik_actions = super()._compose_control(None)
        obs = np.concatenate([obs] + self.ik_actions)
        return obs


class PauseIKToggleEnv(IKToggleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_arm_actions = [np.zeros(self.dof) for _ in range(self.num_arms)]

    def _compose_control(self, rl_action):
        arm_actions = [self.ik_actions[i] if rl_action[i] == 1
                       else self.last_arm_actions[i]
                       for i in range(self.num_arms)]
        self.last_arm_actions = arm_actions
        return arm_actions


class BackupIKToggleEnv(IKToggleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compose_control(self, rl_action):
        arm_actions = [self.ik_actions[i] if rl_action[i] == 1
                       else self.ik_policies[i].default_pose
                       for i in range(self.num_arms)]
        return arm_actions
