import environments
import itertools
import shutil
import stable_baselines3
import time
import wandb

# BEGIN configurable part #

RUN_ID = None #"vwq53k5z"
CHECKPOINT = 10 / 10  # available: 1 to 10 out of 10
NUM_EPISODES = 10
RESOLUTION = 1024

# END configurable part #


if __name__ == "__main__":
    if RUN_ID is None:
        env = environments.TaskEnv(render_mode="human", width=RESOLUTION, height=RESOLUTION, num_arms=2)
    else:
        api = wandb.Api()
        run = api.run(f"nelorth/adlr/{RUN_ID}")
        chkpt_timestep = int(CHECKPOINT * int(run.config['total_timesteps']))
        model_file = run.file(f"checkpoints/rl_model_{chkpt_timestep}_steps.zip").download("/tmp", replace=True)
        shutil.copy(model_file.name, f"policies/{RUN_ID}.zip")

        run.config["env_kwargs"]["render_mode"] = "human"
        run.config["env_kwargs"]["width"] = RESOLUTION
        run.config["env_kwargs"]["height"] = RESOLUTION

        env = getattr(environments, run.config["env_class"])(**run.config["env_kwargs"])
        model = getattr(stable_baselines3, run.config["rl_algo"]).load(f"policies/{RUN_ID}.zip")

    obs, info = env.reset()

    for e in range(NUM_EPISODES):
        tick = time.time()
        for t in itertools.count():
            if RUN_ID is None:
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs)
            obs, reward, terminate, truncate, info = env.step(action)
            print(f"IK states: {tuple(env.ik_policies[i].state.name for i in range(env.num_arms))}")
            print(f"reward: {reward:.4f}")
            if terminate:
                print(
                    f"Episode {e}, Score: {info['scores']}, FPS: {t / (time.time() - tick):.2f}"
                )
                env.reset()
                break

    env.close()
