import os

import torch
import ale_py
import gymnasium as gym

from project.environments import BaseWrapper
from project.agents import DQNAgent
from project.environments.loops import TrainingLoop

PATH_CHECKPOINT = os.path.join("parameters", "model-20042026.pth")
PATH_MEMORY = os.path.join("parameters", "memory-20042026.pth")

FRAMES = 500_000
LR = 2.5e-4
DISCOUNT = 0.99
REPLAY_SIZE = 500_000
ENVIRONMENT = "ALE/Breakout-v5"
TARGET_UPDATE_FREQUENCY = 10_000
BATCH_SIZE = 32
FINAL_EXPLORATION_FRAME = 1_000_000
REPLAY_START_SIZE = 50_000

if __name__ == "__main__":
    gym.register_envs(ale_py)

    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using: {device}")

    env = BaseWrapper.create_environment(ENVIRONMENT)

    agent = DQNAgent(
        train=True,
        lr=LR,
        discount=DISCOUNT,
        replay_size=REPLAY_SIZE,
        n_actions=env.action_space.n,
        target_update_frequency=TARGET_UPDATE_FREQUENCY,
        batch_size=BATCH_SIZE,
        final_exploration_frame=FINAL_EXPLORATION_FRAME,
        replay_start_size=REPLAY_START_SIZE,
        obs_shape=env.observation_space.shape,
        device=device
    )
    if os.path.exists(PATH_CHECKPOINT):
        print("Loading existing checkpoint...")
        agent.load(PATH_CHECKPOINT)
        print("Loaded existing checkpoint!")
    if os.path.exists(PATH_MEMORY):
        print("Loading existing memory...")
        agent.load_memory(PATH_MEMORY)
        print("Loaded existing memory!")

    loop = TrainingLoop(
        env=env,
        agent=agent
    )

    loop.run(FRAMES)

    print("Saving checkpoint and memory...")
    agent.save(PATH_CHECKPOINT)
    agent.save_memory(PATH_MEMORY)
    print("Saved checkpoint and memory!")
