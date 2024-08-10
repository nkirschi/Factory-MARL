import copy
import numpy as np
from typing import Any, Dict, Optional, Tuple
from gymnasium import spaces

from challenge_env.base_env import BaseEnv
from challenge_env.ik_policy import IKPolicy, PolicyState
from dm_control import mjcf

"""
This file contains all custom Gym environment definitions. Their inheritance structure is as follows:

    FactoryManipulationEnv
    ├── ProgressRewardEnv
    │   ├── SingleFullRLProgressRewardEnv
    │   ├── SingleDeltaProgressRewardEnv
    │   ├── AllFullRLProgressRewardEnv
    │   └── AllDeltaProgressRewardEnv
    └──  IKTogglingEnv
        ├── PauseIKToggleEnv
        └── BackupIKToggleEnv
"""


class FactoryManipulationEnv(BaseEnv):
    def __init__(self, **kwargs):
        """
        This class defines the root factory manipulation environment, including custom observation and action spaces,
        and a custom reward function. It inherits from BaseEnv, itself a subclass of gym.Env.

        Parameters
        ----------
        kwargs
            All arguments accepted by BaseEnv.__init__
        """

        super().__init__(**kwargs)

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
        Compute a custom feature representation of the given environment state.

        Parameters
        ----------
        state : np.ndarray
            The environment state as defined in challenge_specifications.STATE_DTYPE

        Returns
        -------
        obs : np.ndarray
            The custom feature representation of the state as expected by the agent
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
        Translate the given agent action into the environment's coordinates.

        Parameters
        ----------
        action : np.ndarray
            The action vector output by the agent

        Returns
        -------
        ctrl : np.ndarray
            The translated control signal as expected by the environment
        """
        action = np.tanh(action)  # squish action into interval (-1, 1)
        bounds = self.joint_limits.copy()  # joint_limits is a property of BaseEnv
        low, high = bounds.T
        ctrl = low + (action + 1.0) * 0.5 * (high - low)
        return ctrl

    def _compose_control(self, action: np.ndarray) -> np.ndarray:
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
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

    def _get_reward(self, state: np.ndarray, action: np.ndarray, info: dict) -> float:
        """
        Compute the reward for a given state-action pair.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment
        action : np.ndarray
            The action taken in the current state
        info : dict
            Additional information about the current state

        Returns
        -------
        reward : float
            The computed reward
        """

        reward = sum(info["scores"]) - sum(self.last_score)
        return reward

    def step(
            self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions.

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Parameters
        ----------
        action : ActType
            An action provided by the agent to update the environment state.

        Returns
        -------
        observation : ObsType
            An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
            An example is a numpy array containing the positions and velocities of the pole in CartPole.
        reward : SupportsFloat
            The reward as a result of taking the action.
        terminated : bool
            Whether the agent reaches the terminal state (as defined under the MDP of the task)
            which can be positive or negative. An example is reaching the goal state or moving into the lava from
            the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
        truncated : bool
            Whether the truncation condition outside the scope of the MDP is satisfied.
            Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
            Can be used to end the episode prematurely before a terminal state is reached.
            If true, the user needs to call :meth:`reset`.
        info : dict
            Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            This might, for instance, contain: metrics that describe the agent's performance state, variables that are
            hidden from observations, or individual reward terms that are combined to produce the total reward.
            In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
            however this is deprecated in favour of returning terminated and truncated variables.
        done : bool, deprecated
            A boolean value for if the episode has ended, in which case further :meth:`step` calls will
            return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
            A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
            a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """

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
            self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        Parameters
        ----------
        seed : int, optional
            The seed that is used to initialize the environment's PRNG (`np_random`).
            If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
            a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
            However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
            If you pass an integer, the PRNG will be reset even if it already exists.
            Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
            Please refer to the minimal example above to see this paradigm in action.
        options : dict, optional
            Additional information to specify how the environment is reset (optional,
            depending on the specific environment)

        Returns
        -------
        observation : ObsType
            Observation of the initial state. This will be an element of :attr:`observation_space`
            (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
        info : dict
            This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
            the ``info`` returned by :meth:`step`.
        """
        state, info = self.reset_sim(seed=seed)
        for ik_policy in self.ik_policies:
            ik_policy.reset()
        obs = self._process_observation(state)
        self.last_score = info["scores"].copy()

        return obs, {}


class ProgressRewardEnv(FactoryManipulationEnv):
    def __init__(
            self,
            gripper_to_closest_cube_reward_factor: float,
            closest_cube_to_bucket_reward_factor: float,
            small_action_norm_reward_factor: float,
            base_reward: float = 0.0,
            **kwargs
    ):
        """
        This class extends the FactoryManipulationEnv class and adds progress-oriented behavioural incentives to the reward function.
        They are calculated based on the distance between the gripper and the cube, and the distance between
        the cube and the bucket.

        Parameters
        ----------
        gripper_to_closest_cube_reward_factor : float
            Weight for the gripper-to-closest-cube incentive.
        closest_cube_to_bucket_reward_factor : float
            Weight for the closest-cube-to-bucket incentive.
        base_reward : float, default 0
            A base reward to avoid large negative rewards.
        kwargs: dict
            All arguments accepted by FactoryManipulationEnv.__init__
        """
        super().__init__(**kwargs)
        self.gripper_to_closest_cube_reward_factor = gripper_to_closest_cube_reward_factor
        self.closest_cube_to_bucket_reward_factor = closest_cube_to_bucket_reward_factor
        self.small_action_norm_reward_factor = small_action_norm_reward_factor
        self.last_gripper_to_closest_cube_dist = [0] * self.num_arms
        self.last_bucket_to_closest_cube_dist = [0] * self.num_arms
        self.base_reward = base_reward

    def _compute_gripper_to_closest_cube_dist(
            self, ik_policy: IKPolicy, last_dist_gripper_to_closest_cube: float
    ) -> Tuple[mjcf.RootElement | None, float, float]:
        """
        Compute the distance between the given gripper and the closest cube in the scene that it is not ignoring.

        Parameters
        ----------
        ik_policy : IKPolicy
            The inverse kinematics policy of the gripper to calculate the distance for.
        last_dist_gripper_to_closest_cube : float
            The previous distance between the gripper and the closest cube.

        Returns
        -------
        The closest cube, its distance to the gripper, and the change in distance to the previous step.
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

    def _compute_bucket_to_closest_cube_dist(
            self, closest_cube: mjcf.RootElement, bucket_pos: float, last_dist_bucket_cube: float
    ) -> Tuple[float, float]:
        """
        Compute the distance between the given cube and the closest bucket.

        Parameters
        ----------
        closest_cube : mjcf.RootElement
            The cube that is closest to the gripper.
        bucket_pos : np.ndarray
            The position of the bucket.
        last_dist_bucket_cube : float
            The previous distance between the bucket and the cube.

        Returns
        -------
        The distance to the bucket and the change in distance to the previous step.
        """
        if closest_cube is None:
            return last_dist_bucket_cube, 0

        cube_pos = self.physics.bind(closest_cube).qpos[:3]
        dist_bucket_cube = np.linalg.norm(cube_pos - bucket_pos)

        return dist_bucket_cube, last_dist_bucket_cube - dist_bucket_cube

    def _get_reward(self, state, action, info) -> float:
        """
        This method overrides the `_get_reward` method from the parent class `FactoryManipulationEnv`.
        The reward is computed based on the base reward, the distance between the gripper and the cube,
        and the distance between the cube and the bucket.
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
        """
        This class extends the ProgressRewardEnv class. The control is defined such that a single arm is fully
        controlled by the RL agent and the rest follows the IK base policy.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by ProgressRewardEnv.__init__
        """
        super().__init__(**kwargs)
        action_dims = self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, action):
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
        """
        ik_actions = super()._compose_control(None)
        return [self._process_action(action)] + ik_actions[1:]


class SingleDeltaProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        """
        This class extends the ProgressRewardEnv class. The control is defined such that all arms follow their IK base
        policy but one arm deviates from it as commanded by the RL agent.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by ProgressRewardEnv.__init__
        """
        super().__init__(**kwargs)
        action_dims = self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, action):
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
        """
        ik_actions = super()._compose_control(None)
        if self.ik_policies[0].state is not PolicyState.IDLE:
            ik_actions[0] += 0.5 * self._process_action(action)
        return ik_actions


class AllFullRLProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        """
        This class extends the ProgressRewardEnv class. The control is defined such that all arms are fully
        controlled by the RL agent and no IK base policy is used at all.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by ProgressRewardEnv.__init__
        """
        super().__init__(**kwargs)
        action_dims = self.num_arms * self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, action):
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
        """
        return [self._process_action(action[self.dof * i:self.dof * (i + 1)]) for i in range(self.num_arms)]


class AllDeltaProgressRewardEnv(ProgressRewardEnv):
    def __init__(self, **kwargs):
        """
        This class extends the ProgressRewardEnv class. The control is defined such that all arms follow their IK base
        policy but deviate from it as commanded by the RL agent.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by ProgressRewardEnv.__init__
        """
        super().__init__(**kwargs)
        action_dims = self.num_arms * self.dof
        self.action_space = spaces.Box(
            low=-np.ones(action_dims),
            high=np.ones(action_dims),
            dtype=np.float32
        )

    def _compose_control(self, action):
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
        """
        ik_actions = super()._compose_control(None)
        for i in range(self.num_arms):
            if self.ik_policies[i].state is not PolicyState.IDLE:
                ik_actions[i] += 0.5 * self._process_action(action[self.dof * i:self.dof * (i + 1)])
        return ik_actions


class IKTogglingEnv(FactoryManipulationEnv):
    def __init__(self, **kwargs):
        """
        This class extends the FactoryManipulationEnv class and redefines the observation space to also include all
        actions proposed by the IK base policies.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by FactoryManipulationEnv.__init__
        """
        super().__init__(**kwargs)

        self.action_space = spaces.MultiDiscrete(self.num_arms * [2])

        obs_dims = self.num_arms * 3 * self.dof + 13 * self.max_num_objects + self.dof * self.num_arms  # add proposed IK actions
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dims),
            high=np.inf * np.ones(obs_dims),
            dtype=np.float32,
        )

    def _process_observation(self, state: np.ndarray) -> np.ndarray:
        """
        Compute a custom feature representation of the given environment state.

        Parameters
        ----------
        state : np.ndarray
            The environment state as defined in challenge_specifications.STATE_DTYPE

        Returns
        -------
        obs : np.ndarray
            The custom feature representation of the state as expected by the agent
        """
        obs = super()._process_observation(state)
        self.ik_actions = super()._compose_control(None)
        obs = np.concatenate([obs] + self.ik_actions)
        return obs


class PauseIKToggleEnv(IKTogglingEnv):
    def __init__(self, **kwargs):
        """
        This class extends the IKTogglingEnv class. The control is defined such the RL agent decides for each arm
        whether it should follow its IK base policy or freeze at the current position.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by IKTogglingEnv.__init__
        """
        super().__init__(**kwargs)
        self.last_arm_actions = [np.zeros(self.dof) for _ in range(self.num_arms)]

    def _compose_control(self, action):
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
        """
        arm_actions = [self.ik_actions[i] if action[i] == 1
                       else self.last_arm_actions[i]
                       for i in range(self.num_arms)]
        self.last_arm_actions = arm_actions
        return arm_actions


class BackupIKToggleEnv(IKTogglingEnv):
    def __init__(self, **kwargs):
        """
        This class extends the IKTogglingEnv class. The control is defined such the RL agent decides for each arm
        whether it should follow its IK base policy or retreat to a safe position.

        Parameters
        ----------
        kwargs: dict
            All arguments accepted by IKTogglingEnv.__init__
        """
        super().__init__(**kwargs)

    def _compose_control(self, action):
        """
        Compose the final robot control vector from the given processed action

        Parameters
        ----------
        action : np.ndarray
            The already processed action from the agent as output by _process_action

        Returns
        -------
        arm_actions : np.ndarray
            The composed control directives for the robots
        """
        arm_actions = [self.ik_actions[i] if action[i] == 1
                       else self.ik_policies[i].default_pose
                       for i in range(self.num_arms)]
        return arm_actions
