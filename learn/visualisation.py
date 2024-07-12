import environments
import itertools
import shutil
import stable_baselines3
import time
import wandb

# BEGIN configurable part #

RUN_ID = "vafxihae"
CHECKPOINT = 10 / 10  # available: 1 to 10 out of 10
NUM_EPISODES = 10

# END configurable part #


api = wandb.Api()
run = api.run(f"nelorth/adlr/{RUN_ID}")
chkpt_timestep = int(CHECKPOINT * int(run.config['total_timesteps']))
model_file = run.file(f"checkpoints/rl_model_{chkpt_timestep}_steps.zip").download("/tmp", replace=True)
shutil.copy(model_file.name, f"policies/{RUN_ID}.zip")

env = getattr(environments, run.config["env_class"])(**run.config["env_kwargs"])
model = getattr(stable_baselines3, run.config["rl_algo"]).load(f"policies/{RUN_ID}.zip")

obs, info = env.reset()

for e in range(NUM_EPISODES):
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
