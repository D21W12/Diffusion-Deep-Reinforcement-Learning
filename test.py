import os

import torch
import ale_py
import gymnasium as gym

from project.agents import DDQNAgent, DQNAgent
from project.environments import BaseWrapper
from project.environments.loops import TestingLoop

PATH_CHECKPOINT = os.path.join("parameters", "model-22032026.pth")
ENVIRONMENT = "ALE/Breakout-v5"


if __name__ == "__main__":

    gym.register_envs(ale_py)
    env = BaseWrapper.create_environment(ENVIRONMENT, render_mode="human")

    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using: {device}")

    agent = DQNAgent(
        train=False,
        n_actions=env.action_space.n,
        device=device,
    )
    if os.path.exists(PATH_CHECKPOINT):
        print("Loading existing checkpoint...")
        agent.load(PATH_CHECKPOINT)
        print("Loaded existing checkpoint!")

    loop = TestingLoop(
        env=env,
        agent=agent
    )
    loop.run()
