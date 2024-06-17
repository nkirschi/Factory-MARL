import time
from stable_baselines3 import PPO, SAC
from environments import *

env = SingleDeltaEnv(1, 1, render_mode="human")
obs, info = env.reset()
num_episodes = 10

model = SAC.load("policies/SAC_1_1")

frames = []
for e in range(num_episodes):
    tick = time.time()
    t = 0
    while True:
        t += 1
        action, _states = model.predict(obs)
        obs, reward, terminate, truncate, info = env.step(action)
        print("action:", action)
        print("reward:", reward)
        if terminate:
            print(
                f"Episode {e}, Score: {info['scores']}, FPS: {t / (time.time() - tick):.2f}"
            )
            env.reset()
            break

env.close()
