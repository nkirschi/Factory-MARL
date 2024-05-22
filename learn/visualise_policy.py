import time
from stable_baselines3 import PPO
from single_delta_env import SingleDeltaEnv

env = SingleDeltaEnv(render_mode="human")
obs, info = env.reset()
num_episodes = 10

model = PPO.load("single_delta_policy")

frames = []
for e in range(num_episodes):
    tick = time.time()
    t = 0
    while True:
        t += 1
        action = model.predict(obs)[0]
        obs, reward, terminate, truncate, info = env.step(action)
        if terminate:
            print(
                f"Episode {e}, Score: {info['scores']}, FPS: {t / (time.time() - tick):.2f}"
            )
            env.reset()
            break

env.close()
