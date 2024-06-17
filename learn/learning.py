from gymnasium.wrappers import NormalizeReward
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, ProgressBarCallback
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

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        env = self.training_env
        try:
            self.logger.record("ep_score_last", env.last_score)
        except AttributeError:
            pass
        return True


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "osmesa"
    CONFIG = {
        "num_envs": 4,
        "env_class": SingleDeltaEnv,
        "notes": "SAC with true score reward signal",
        "rl_algo": SAC,
        "total_timesteps": int(1e6),
        "log_interval": 5,  # for on-policy algos: #steps, for off-policy algos: #episodes
        "chkpt_interval": int(1e6 / 10),
        "policy_type": "MlpPolicy",
        "score_weight": 1,
        "norm_penalty_weight": 0.1
    }


    def make_env():
        env = CONFIG["env_class"](CONFIG["score_weight"],
                                  CONFIG["norm_penalty_weight"],
                                  render_mode="rgb_array")
        return Monitor(env)


    wandb.login(key="f4cdba55e14578117b20251fd078294ca09d974d", verify=True)
    run = wandb.init(project="adlr",
                     notes=CONFIG["notes"],
                     config=CONFIG,
                     sync_tensorboard=True,
                     monitor_gym=True,
                     save_code=True)
    env = make_vec_env(make_env,
                       n_envs=CONFIG["num_envs"],
                       vec_env_cls=SubprocVecEnv)
    env.reset()
    env = VecVideoRecorder(env,
                           video_folder=f"videos/{run.id}",
                           record_video_trigger=lambda x: x % (CONFIG["chkpt_interval"] / CONFIG["num_envs"]) == 0,
                           video_length=100)
    model = CONFIG["rl_algo"](CONFIG["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=CONFIG["total_timesteps"],
                log_interval=CONFIG["log_interval"],
                callback=[  # ProgressBarCallback(),
                    CheckpointCallback(save_freq=CONFIG["chkpt_freq"],
                                       save_path=f"policies/{run.id}"),
                    WandbCallback(model_save_freq=CONFIG["chkpt_freq"],
                                  model_save_path=f"policies/{run.id}",
                                  verbose=2)])
    run.finish()
