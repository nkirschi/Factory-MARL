from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder
from environments import *
from wandb.integration.sb3 import WandbCallback

import os
import wandb

if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "osmesa"
    CONFIG = {
        "num_envs": 8,
        "env_class": SingleDeltaEnvWithNormPenalty,
        "run_name": "single_delta_policy_with_norm_penalty",
        "rl_algo": PPO,
        "total_timesteps": int(1e5),
        "policy_type": "MlpPolicy",
        "chkpt_freq": 16384,
    }
    wandb.login(key="f4cdba55e14578117b20251fd078294ca09d974d", verify=True)
    run = wandb.init(project="adlr",
                     name=CONFIG["run_name"],
                     config=CONFIG,
                     sync_tensorboard=True,
                     monitor_gym=True,
                     save_code=True)
    env = make_vec_env(lambda: Monitor(CONFIG["env_class"](render_mode="rgb_array")),
                       n_envs=CONFIG["num_envs"],
                       vec_env_cls=SubprocVecEnv)
    env.reset()
    env = VecVideoRecorder(env,
                           video_folder=f"videos/{run.id}",
                           record_video_trigger=lambda x: x % CONFIG["chkpt_freq"] == 0,
                           video_length=100)
    model = CONFIG["rl_algo"](CONFIG["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=CONFIG["total_timesteps"],
                callback=[WandbCallback(
                    model_save_freq=CONFIG["chkpt_freq"],
                    model_save_path=f"policies/{run.id}",
                    verbose=2,
                ), ProgressBarCallback()])
    run.finish()
