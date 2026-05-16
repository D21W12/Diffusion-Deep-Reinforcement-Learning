import argparse
import os
import random

import ale_py
import gymnasium as gym
import numpy as np
import torch

from project.agents import DDQNAgent
from project.environments import BaseWrapper
from project.environments.loops import TrainingLoop
from project.train.config import DDQNTrainingTrainingConfig


def train_ddqn(
        checkpoint_path: str,
        device: str,
        epochs: int,
        memory_checkpoint_path: str | None = None,
        interval: int | None = None,
        tag: str = "",
) -> None:

    config = DDQNTrainingTrainingConfig(
        checkpoint_path=checkpoint_path,
        memory_checkpoint_path=memory_checkpoint_path,
        device=device,
    )

    # Setting the seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    def checkpoint():
        print("Saving checkpoint...")
        agent.save(config.checkpoint_path)
        print("Saved checkpoint!")
        if memory_checkpoint_path is not None:
            print("Saving memory...")
            agent.save_memory(config.memory_checkpoint_path)
            print("Saved memory!")

    gym.register_envs(ale_py)

    env = BaseWrapper.create_environment(config.environment)

    agent = DDQNAgent(
        train=True,
        lr=config.lr,
        discount=config.discount,
        replay_size=config.replay_size,
        n_actions=env.action_space.n,
        target_update_frequency=config.target_update_frequency,
        batch_size=config.batch_size,
        final_exploration_frame=config.final_exploration_frame,
        replay_start_size=config.replay_start_size,
        device=config.device
    )
    if os.path.exists(config.checkpoint_path):
        print("Loading existing checkpoint...")
        agent.load(config.checkpoint_path)
        print("Loaded existing checkpoint!")
    if memory_checkpoint_path and os.path.exists(config.memory_checkpoint_path):
        print("Loading existing memory...")
        agent.load_memory(config.memory_checkpoint_path)
        print("Loaded existing memory!")

    loop = TrainingLoop(
        env=env,
        agent=agent,
        lr=config.lr,
        tag=tag,
    )

    loop.run(
        frames=epochs,
        checkpoint=interval,
        checkpoint_callback=checkpoint,
    )

    checkpoint()


def main():

    parser = argparse.ArgumentParser(
        prog='Project train (Double-DQN)',
        description='This program manages train for my BSc Thesis project (Double-DQN).',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-c', '--checkpoint', required=True, type=str)
    parser.add_argument('-i', '--interval', type=int)

    parser.add_argument('-d', '--device', default='cpu')

    parser.add_argument('-e', '--epochs', required=True, type=int)

    parser.add_argument('--memory', required=True)

    parser.add_argument('-t', '--tag', required=True)

    args = parser.parse_args()

    train_ddqn(
        checkpoint_path=args.checkpoint,
        device=args.device,
        memory_checkpoint_path=args.memory,
        epochs=args.epochs,
        interval=args.interval,
        tag=args.tag,
    )

if __name__ == "__main__":
    main()
