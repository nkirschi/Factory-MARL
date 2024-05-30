from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from environments import *

if __name__ == "__main__":
    NUM_ENVS = 8
    TOTAL_TIMESTEPS = int(1e7)

    env = make_vec_env(lambda: SingleDeltaEnvWithNormPenalty(render_mode="rgb_array"),
                       n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    env.reset()
    model = PPO("MlpPolicy", env, verbose=1)
    #model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save("policies/single_delta_policy_with_norm_penalty")
