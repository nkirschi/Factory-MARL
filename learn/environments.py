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
    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        width=1024,
        height=1024
    ):
        super().__init__(render_mode=render_mode, seed=seed, width=width, height=height)
        self.old_min_dist_0 = 0
        self.old_min_dist_1 = 0
        self.old_min_bucket_dist_0 = 0
        self.old_min_bucket_dist_1 = 0

    def _compose_control(self, rl_action):
        ik_action0, ik_action1 = super()._compose_control(rl_action)
        action_arm0 = ik_action0 + 0.5 * self._process_action(rl_action)
        action_arm1 = ik_action1
        return action_arm0, action_arm1

    def _compute_min_distance(self, ik_policy, old_min_dist):
        position = self.physics.named.data.site_xpos[ik_policy._gripper_site]

        candidates = copy.copy(self.task_manager._in_scene)
        if ik_policy.ignore_object is not None:
            if ik_policy.ignore_object in candidates:
                candidates.remove(ik_policy.ignore_object)

        if len(candidates) == 0:
            return None, old_min_dist, 0

        pos = self.physics.bind(candidates).qpos.reshape(-1, 7)[:, :3]
        dists = np.linalg.norm(
            pos - position,
            axis=1,
            )

        diff = old_min_dist - np.min(dists)

        closest_object = candidates[np.argmin(dists)]
        return closest_object, np.min(dists), diff

    def _compute_bucket_distance_cube(self, min_object, bucket_pos, old_min_bucket_dist):
        # position = self.physics.named.data.site_xpos[ik_policy._gripper_site]
        if min_object is None:
            return old_min_bucket_dist, 0

        pos = self.physics.bind(min_object).qpos[:3]
        dist = np.linalg.norm(pos - bucket_pos)
        diff = old_min_bucket_dist - dist

        return dist, diff

    def _get_reward(self, state, action, info) -> float:
        # Params
        alpha_1 = .1  # Penalty weight for distance to target
        alpha_2 = .2  # Penalty weight for distance to bucket
        reward = sum(info["scores"]) - sum(self.last_score)
        reward *= 10

        old_min_dist = self.old_min_dist_0
        min_object, min_dist, diff = self._compute_min_distance(self.ik_policy0, old_min_dist)
        self.old_min_dist_0 = min_dist

        old_min_dist = self.old_min_dist_1
        min_object2, min_dist2, diff2 = self._compute_min_distance(self.ik_policy1, old_min_dist)
        self.old_min_dist_1 = min_dist2

        old_bucket_dist = self.old_min_bucket_dist_0
        bucket_0_pos = [0.9, 0.7, 1.05]
        bucket_dist, diff_bucket = self._compute_bucket_distance_cube(min_object, bucket_0_pos,
                                                                      old_bucket_dist)
        self.old_min_bucket_dist_0 = bucket_dist

        old_bucket_dist = self.old_min_bucket_dist_1
        bucket_1_pos = [-0.9, 0.7, 1.05]
        bucket_dist2, diff_bucket2 = self._compute_bucket_distance_cube(min_object2, bucket_1_pos,
                                                                        old_bucket_dist)
        self.old_min_bucket_dist_1 = bucket_dist2

        # print(f"dist: {min_dist}, bucket_dist: {bucket_dist}")
        # print("diff:", diff, diff2, diff_bucket, diff_bucket2)

        reward = reward + (alpha_1 * diff + alpha_2 * diff2)  # Add penalty for distance to target
        reward = reward + (alpha_1 * diff_bucket + alpha_2 * diff_bucket2)  # Add penalty for distance to bucket
        # print("reward:", reward)
        return reward

