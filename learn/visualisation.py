import itertools
import time
from stable_baselines3 import PPO, SAC
from environments import *

# env = TaskEnv(render_mode="human")
# env = SingleFullRLEnv(render_mode="human")
# env = SingleDeltaEnv(1, 1, render_mode="human")
env = SingleFullRLProgressRewardEnv(gripper_to_closest_cube_reward_factor=0.1,
                                    closest_cube_to_bucket_reward_factor=0.1,
                                    base_reward=0,
                                    small_action_norm_reward_factor=0,
                                    render_mode="human")
obs, info = env.reset()
num_episodes = 10

model = SAC.load("policies/ProgressRewSAC.zip")

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
