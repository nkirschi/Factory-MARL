from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from single_delta_env import SingleDeltaEnv

num_envs = 8
env = make_vec_env(lambda: SingleDeltaEnv(render_mode="rgb_array"), n_envs=num_envs)
env.reset()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000, progress_bar=False)
model.save("single_delta_policy")
