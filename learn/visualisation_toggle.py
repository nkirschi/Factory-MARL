import itertools
import time
from stable_baselines3 import PPO, SAC
from environments import *

env = BackupIKToggleEnv(render_mode="human")
obs, info = env.reset()
num_episodes = 10

model = PPO.load("policies/Backup.zip")

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
