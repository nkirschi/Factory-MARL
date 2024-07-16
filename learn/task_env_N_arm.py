# Example for a custom task definition

import numpy as np
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces

from challenge_env.base_env import BaseEnv
from challenge_env.ik_policy import IKPolicy


class TaskEnv(BaseEnv):
    def __init__(
        self,
        num_arms = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(render_mode=render_mode, seed=seed, num_arms=num_arms)
        self.num_arms = num_arms

        self.ik_indices = [0, 3]
        self.ik_policies = []

        self.task_manager._spawn_pos[1] += 0.2 # slighly adjust the spawn position of the objects

        for i in range(self.num_arms):
            if i in self.ik_indices:
                self.ik_policies.append(IKPolicy(self, arm_id=i, bucket_idx = i%2))

        # Example custom observation and action spaces:
        obs_dims = self.num_arms * 2 * self.dof + 13 * self.max_num_objects
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dims),
            high=np.inf * np.ones(obs_dims),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-np.ones(self.dof), high=np.ones(self.dof), dtype=np.float32
        )

    def _process_observation(self, state: np.ndarray) -> np.ndarray:
        dof_measurements = np.concatenate(
            [
                [np.concatenate([state[f"qpos_player{i}"].flatten(),
                state[f"qvel_player{i}"].flatten()], axis=0)]
                for i in range(self.num_arms)
            ], axis=0
        ).flatten()
        object_states = np.concatenate(
            [state["object_poses"].flatten(), state["object_vels"].flatten()]
        )
        obs = np.concatenate([dof_measurements, object_states])
        return obs

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, -1.0, 1.0)
        bounds = self.joint_limits.copy()  # joint_limits is a property of BaseEnv
        low, high = bounds.T
        ctrl = low + (action + 1.0) * 0.5 * (high - low)
        return ctrl

    def _get_reward(self, state, info) -> float:
        # Your custom reward function goes here
        reward = sum(info["scores"]) - sum(self.last_score)
        return reward

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        actions = {}
        for i in range(len(self.ik_indices)):
            actions[f"action_arm{self.ik_indices[i]}"] = self.ik_policies[i].act()

        state, terminate, info = self.step_sim(
            **actions
        )

        reward = self._get_reward(state, info)
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
        [ik.reset() for ik in self.ik_policies]
        obs = self._process_observation(state)
        self.last_score = info["scores"]

        return obs, {}


if __name__ == "__main__":
    import time

    render_mode = "human"  # "human" or "rgb_array"

    env = TaskEnv(render_mode=render_mode)
    env.reset()
    num_episodes = 10

    frames = []
    for e in range(num_episodes):
        tick = time.time()
        t = 0
        while True:
            t += 1
            action = env.action_space.sample() * 0.0
            observation, reward, terminate, truncate, info = env.step(action)
            if terminate:
                print(
                    f"Episode {e}, Score: {info['scores']}, FPS: {t / (time.time() - tick):.2f}"
                )
                env.reset()
                break

    env.close()
