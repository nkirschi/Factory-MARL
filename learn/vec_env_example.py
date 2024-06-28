# example of using gymnasium.vector to create a vectorized environment
# for parallelizing training of agents
import gymnasium as gym
from environments import TaskEnv, TaskEnvWithDistancePenalty
import time


def make_env_f():
    return TaskEnvWithDistancePenalty(render_mode="human", distance_gripper_reward_factor=0.1, distance_bucket_reward_factor=0.2)


if __name__ == "__main__":
    num_envs = 1  # number of parallel environments
    env = gym.vector.AsyncVectorEnv(
        [
            *[make_env_f for _ in range(num_envs)],
        ]
    )

    env.reset()
    tick = time.time()
    episodes_finished = 0

    for t in range(1000):
        action = env.action_space.sample() * 0.0
        obs, reward, term, trunc, info = env.step(action)
        # terminated episodes are reset automatically by the vector env
        episodes_finished += sum(term)
        print(
            f"Step {t}, Episodes done: {episodes_finished},  FPS: {(t * num_envs) / (time.time() - tick):.2f}",
            end="\r",
        )
