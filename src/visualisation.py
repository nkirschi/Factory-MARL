import environments
import itertools
import json
import numpy
import os
import shutil
import stable_baselines3
import time
import tqdm
import wandb


"""
This file contains a visualisation and evaluation script.
"""


# BEGIN configurable part #

RUN_ID = "xfwgqibb" # None  # best run: "y6lp1j7k"
USE_CACHED = False
CHECKPOINT = 10 / 10  # available: 1 to 10 out of 10
NUM_EPISODES = 10
RENDER_MODE = "human"  # either "human" or "rgb_array"
RESOLUTION = 1024
VERBOSE = True

# END configurable part #


if __name__ == "__main__":
    if RUN_ID is None:
        env = environments.FactoryManipulationEnv(render_mode=RENDER_MODE, width=RESOLUTION, height=RESOLUTION, num_arms=4)
    else:
        if not os.path.isfile(f"../runs/{RUN_ID}.json"):
            wandb.login(key="f4cdba55e14578117b20251fd078294ca09d974d", verify=True)
            api = wandb.Api()
            run = api.run(f"nelorth/adlr/{RUN_ID}")
            os.makedirs("../runs/", exist_ok=True)
            with open(f"../runs/{RUN_ID}.json", "w", encoding="utf-8") as f:
                json.dump(run.config, f, ensure_ascii=False, indent=4)
            chkpt_timestep = int(CHECKPOINT * int(run.config['total_timesteps']))
            model_file = run.file(f"checkpoints/rl_model_{chkpt_timestep}_steps.zip").download("/tmp", replace=True)
            shutil.copy(model_file.name, f"../runs/{RUN_ID}.zip")

        with open(f"../runs/{RUN_ID}.json") as f:
            config = json.load(f)
            config["env_kwargs"]["render_mode"] = RENDER_MODE
            config["env_kwargs"]["width"] = RESOLUTION
            config["env_kwargs"]["height"] = RESOLUTION

        env = getattr(environments, config["env_class"])(**config["env_kwargs"])
        model = getattr(stable_baselines3, config["rl_algo"]).load(f"../runs/{RUN_ID}.zip")

    obs, info = env.reset()
    ep_lens = []
    ep_scores = []

    for e in range(NUM_EPISODES) if VERBOSE else tqdm.tqdm(range(NUM_EPISODES)):
        tick = time.time()
        for t in itertools.count():
            if RUN_ID is None:
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs)
            obs, reward, terminate, truncate, info = env.step(action)
            if VERBOSE:
                print(f"IK states: {tuple(env.ik_policies[i].state.name for i in range(env.num_arms))}")
                print(f"action: {action}")
                print(f"reward: {reward:.4f}")
            if terminate:
                if VERBOSE:
                    print(f"Episode {e}, Score: {info['scores']}, FPS: {t / (time.time() - tick):.2f}")
                ep_scores.append(info['scores'])
                ep_lens.append(t)
                env.reset()
                break

    print("avg episode scores:", numpy.array(ep_scores).mean(axis=0))
    print("avg episode lengths", numpy.array(ep_lens).mean())

    env.close()
