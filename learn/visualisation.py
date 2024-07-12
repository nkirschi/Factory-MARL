import itertools
import time
from stable_baselines3 import PPO, SAC
from environments import *

# BEGIN CONFIGURABLE PART #

rl_algo = PPO
env_class = BackupIKToggleEnv
env_kwargs = dict(
    render_mode="human",
    width=1024,
    height=1024,
    #gripper_to_closest_cube_reward_factor=0.1,
    #closest_cube_to_bucket_reward_factor=0.1,
    #base_reward=0,
    #small_action_norm_reward_factor=0,
)

# END CONFIGURABLE PART #


env = env_class(**env_kwargs)
obs, info = env.reset()
num_episodes = 10

model = rl_algo.load("policies/BackupIKToggleEnv_10000000_steps.zip")

for e in range(num_episodes):
    tick = time.time()
    for t in itertools.count():
        action, _states = model.predict(obs)
        obs, reward, terminate, truncate, info = env.step(action)
        print(f"IK states: ({env.ik_policy1.state.name}, {env.ik_policy0.state.name})")
        print(f"reward: {reward:.4f}")
        if terminate:
            print(
                f"Episode {e}, Score: {info['scores']}, FPS: {t / (time.time() - tick):.2f}"
            )
            env.reset()
            break

env.close()
