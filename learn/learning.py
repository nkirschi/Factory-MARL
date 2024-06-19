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


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "osmesa"
    CONFIG = {
        "num_envs": 4,
        "env_class": SingleDeltaEnv,
        "notes": "SAC with true score reward signal",  # adjust this before every run
        "rl_algo": SAC,
        "total_timesteps": int(1e6),
        "log_interval": 5,  # for on-policy algos: #steps, for off-policy algos: #episodes
        "chkpt_interval": int(1e6 / 10),
        "policy_type": "MlpPolicy",
        "score_weight": 1,
        "norm_penalty_weight": 0
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
    os.makedirs("policies_sb3", exist_ok=True)
    wandb.save(f"policies_sb3/{run.id}/*")
    os.makedirs("policies_wandb", exist_ok=True)
    wandb.save(f"policies_wandb/{run.id}/*")
    env = make_vec_env(make_env,
                       n_envs=CONFIG["num_envs"],
                       vec_env_cls=SubprocVecEnv)
    env.reset()
    env = VecVideoRecorder(env,
                           video_folder=f"videos/{run.id}",
                           record_video_trigger=lambda x: x % (CONFIG["chkpt_interval"] // CONFIG["num_envs"]) == 0,
                           video_length=100)
    model = CONFIG["rl_algo"](CONFIG["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=CONFIG["total_timesteps"],
                log_interval=CONFIG["log_interval"],
                callback=[  # ProgressBarCallback(),
                    CheckpointCallback(save_freq=CONFIG["chkpt_interval"] // CONFIG["num_envs"],
                                       save_path=f"policies_sb3/{run.id}"),
                    WandbCallback(model_save_freq=CONFIG["chkpt_interval"] // CONFIG["num_envs"],
                                  model_save_path=f"policies_wandb/{run.id}",
                                  verbose=2),
                    AdditionalMetricsCallback()])

    run.finish()
