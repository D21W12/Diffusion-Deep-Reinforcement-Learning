import os

import ale_py
import gymnasium as gym

from project.agents import DDQNAgent
from project.environments import BaseWrapper
from project.environments.loops import TestingLoop


if __name__ == "__main__":

    PATH = os.path.join("parameters", "model-ddqn-2m-18032026.pth")

    gym.register_envs(ale_py)

    env = BaseWrapper.create_environment('ALE/Breakout-v5', render_mode="human")

    agent = DDQNAgent(
        train=False,
        n_actions=4,
        obs_shape=env.observation_space.shape
    )
    if os.path.exists(PATH):
        agent.load(PATH)
        print("Loaded existing parameters")

    loop = TestingLoop(
        env=env,
        agent=agent
    )
    loop.run()
