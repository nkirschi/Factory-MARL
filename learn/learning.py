from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from environments import *
from wandb.integration.sb3 import WandbCallback

import os
import wandb


class AdditionalMetricsCallback(BaseCallback):
    """
    Custom callback for plotting additional metrics in tensorboard.
    """

    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        try:
            scores = self.training_env.get_attr("last_score")
            avg_total_score = sum(map(sum, scores)) / len(scores)
            self.logger.record("rollout/ep_score_last", avg_total_score)
        except AttributeError:
            raise AssertionError("Property last_score not present in env object", self.training_env)
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
    CONFIG = {
        "num_envs": 8,
        "env_class": SingleDeltaProgressRewardEnv,
        "env_kwargs": {
            "gripper_to_closest_cube_reward_factor": 0.1,
            "closest_cube_to_bucket_reward_factor": 0.1,
            "small_action_norm_reward_factor": 0,
            "base_reward": 0
        },
        "notes": "Nonnegative progress reward with PPO",  # adjust this before every run
        "rl_algo": PPO,
        "total_timesteps": int(1e6),
        "log_interval": 1,  # for on-policy algos: #steps, for off-policy algos: #episodes
        "chkpt_interval": int(1e6 / 10),
        "policy_type": "MlpPolicy",
    }

    # os.environ["MUJOCO_GL"] = "osmesa"
    wandb.login(key="f4cdba55e14578117b20251fd078294ca09d974d", verify=True)
    run = wandb.init(project="adlr",
                     notes=CONFIG["notes"],
                     config=CONFIG,
                     sync_tensorboard=True,
                     monitor_gym=True,
                     save_code=True)
    env = make_vec_env(lambda: Monitor(CONFIG["env_class"](render_mode="rgb_array", **CONFIG["env_kwargs"])),
                       n_envs=CONFIG["num_envs"],
                       vec_env_cls=SubprocVecEnv)
    env.reset()
    # env = VecVideoRecorder(env,
    #                        video_folder=f"runs/{run.id}/videos",
    #                        record_video_trigger=lambda x: x % (CONFIG["chkpt_interval"] // CONFIG["num_envs"]) == 0,
    #                        video_length=100)
    model = CONFIG["rl_algo"](CONFIG["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}/tensorboard")
    model.learn(total_timesteps=CONFIG["total_timesteps"],
                log_interval=CONFIG["log_interval"],
                callback=[  # ProgressBarCallback(),
                    SyncifiedCheckpointCallback(save_freq=CONFIG["chkpt_interval"] // CONFIG["num_envs"],
                                                save_path=f"runs/{run.id}/checkpoints"),
                    WandbCallback(model_save_freq=CONFIG["chkpt_interval"] // CONFIG["num_envs"],
                                  model_save_path=f"runs/{run.id}/checkpoints",
                                  verbose=2),
                    AdditionalMetricsCallback()])

    run.finish()
