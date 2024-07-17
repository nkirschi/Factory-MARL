from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import environments
import os
import stable_baselines3
import wandb


class AdditionalMetricsCallback(BaseCallback):
    """
    Custom callback for plotting additional metrics in tensorboard.
    """

    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        try:
            score_history = self.training_env.get_attr("ep_score_history")
            if all(score_history):  # no env has empty history
                for arm_id in range(self.training_env.get_attr("num_arms")[0]):
                    arm_scores = [score_history[env_id][-t][arm_id]
                                  for env_id in range(len(score_history))
                                  for t in range(min(100, len(score_history[env_id])))]
                    self.logger.record(f"rollout/ep_score_mean_arm{arm_id}", sum(arm_scores) / len(arm_scores))
        except AttributeError:
            raise AssertionError("Property ep_score_history not present in env object", self.training_env)
        return True


class SyncifiedCheckpointCallback(CheckpointCallback):
    def __int__(self, **kwargs):
        super().__init__(**kwargs)

    def _on_step(self) -> bool:
        alive = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            wandb.save(f"{self.save_path}/*", policy="now", base_path="/".join(self.save_path.split("/")[:-1]))
        return alive


if __name__ == "__main__":
    CONFIG = dict(
        num_envs=8,
        env_class="BackupIKToggleEnv",
        env_kwargs={
            "num_arms": 4,
            "render_mode": "rgb_array",
            "seed": 42,
            "initial_conveyor_speed": 0.1,
            "conveyor_acceleration": 0.001,
            "pt_time": 0.2,
            "force_contact_threshold": 200.0,
            "max_num_objects": 10,
            "control_frequency": 10,
            "spawn_freq": 1 / 10,
            "spawn_freq_increase": 1.001,
            # gripper_to_closest_cube_reward_factor=0.1,
            # closest_cube_to_bucket_reward_factor=0.1,
            # small_action_norm_reward_factor=0,
            # base_reward=0
        },
        notes="CHANGEME!",  # adjust this before every run
        rl_algo="PPO",
        total_timesteps=int(5e6),
        policy_type="MlpPolicy",
        policy_kwargs={
            "net_arch": [128, 128]
        },
        discount_factor=0.99,
    )

    # os.environ["MUJOCO_GL"] = "osmesa"
    wandb.login(key="f4cdba55e14578117b20251fd078294ca09d974d", verify=True)
    run = wandb.init(project="adlr",
                     notes=CONFIG["notes"],
                     config=CONFIG,
                     sync_tensorboard=True,
                     monitor_gym=True,
                     save_code=True)
    env = make_vec_env(lambda: Monitor(getattr(environments, CONFIG["env_class"])(**CONFIG["env_kwargs"])),
                       n_envs=CONFIG["num_envs"],
                       vec_env_cls=SubprocVecEnv)
    env.reset()
    # env = VecVideoRecorder(env,
    #                        video_folder=f"runs/{run.id}/videos",
    #                        record_video_trigger=lambda x: x % (CONFIG["chkpt_interval"] // CONFIG["num_envs"]) == 0,
    #                        video_length=100)
    model = getattr(stable_baselines3, CONFIG["rl_algo"])(policy=CONFIG["policy_type"],
                                                          policy_kwargs=CONFIG["policy_kwargs"],
                                                          env=env,
                                                          gamma=CONFIG["discount_factor"],
                                                          tensorboard_log=f"runs/{run.id}/tensorboard",
                                                          verbose=1)
    model.learn(total_timesteps=CONFIG["total_timesteps"],
                callback=[SyncifiedCheckpointCallback(save_freq=CONFIG["total_timesteps"] // 10 // CONFIG["num_envs"],
                                                      save_path=f"runs/{run.id}/checkpoints"),
                          WandbCallback(model_save_freq=CONFIG["total_timesteps"] // 10 // CONFIG["num_envs"],
                                        model_save_path=f"runs/{run.id}/checkpoints",
                                        verbose=2),
                          AdditionalMetricsCallback()])

    run.finish()
