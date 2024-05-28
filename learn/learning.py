from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environments import *


NUM_ENVS = 8
TOTAL_TIMESTEPS = int(1e6)

env = make_vec_env(lambda: SingleDeltaEnvWithNormPenalty(render_mode="rgb_array"), n_envs=NUM_ENVS)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
model.save("policies/single_delta_policy_with_norm_penalty")
