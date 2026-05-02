import os

import ale_py
import gymnasium as gym
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from project.environments import BaseWrapper
from project.agents import DQNAgent
from project.environments.loops import TrainingLoop
from project.models import EDMEvelynn
from project.training.config import DQNConfig, DiffusionConfig
from project.util.data import ReplayMemoryData


def train_dqn(
        checkpoint_path: str,
        memory_checkpoint_path: str,
        device: str,
        *args,
        **kwargs,
) -> None:

    config = DQNConfig(
        checkpoint_path=checkpoint_path,
        memory_checkpoint_path=memory_checkpoint_path,
        device=device,
        *args,
        **kwargs
    )

    def checkpoint():
        print("Saving checkpoint and memory...")
        agent.save(config.checkpoint_path)
        agent.save_memory(config.memory_checkpoint_path)
        print("Saved checkpoint and memory!")

    gym.register_envs(ale_py)

    env = BaseWrapper.create_environment(config.environment)

    agent = DQNAgent(
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
    if os.path.exists(config.memory_checkpoint_path):
        print("Loading existing memory...")
        agent.load_memory(config.memory_checkpoint_path)
        print("Loaded existing memory!")

    loop = TrainingLoop(
        env=env,
        agent=agent
    )

    loop.run(
        frames=config.epochs,
        checkpoint=config.checkpoint,
        checkpoint_callback=checkpoint,
    )

    checkpoint()


def train_diffusion(
        data_path: str,
        checkpoint_path: str,
        device: str,
        epochs: int,
        network: str,
        *args,
        **kwargs,
) -> None:

    config = DiffusionConfig(
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        device=device,
        epochs=epochs,
        network=network,
        *args,
        **kwargs
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    data = ReplayMemoryData(
        memory=config.data_path,
        transform=transform,
    )
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True)

    model = EDMEvelynn(
        img_resolution=config.resolution,
        img_channels=config.in_channels,
        start_channels=config.start_channels,
        channel_mult=config.channel_multipliers,
        num_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        batch_size=config.batch_size,
        lr=config.lr,
        network=config.network
    ).to(config.device)

    if os.path.exists(config.checkpoint_path):
        print("Loading checkpoint...")
        model.load_checkpoint(config.checkpoint_path)
        print("Loaded checkpoint!")

    model.train(config.epochs, loader)

    print("Saving checkpoint...")
    model.save_checkpoint(config.checkpoint_path)
    print("Checkpoint saved!")


def train(
        model: str,
        *args,
        **kwargs,
) -> None:
    if model in ["dqn", "rl"]:
        train_dqn(*args, **kwargs)
    elif model in ["diffusion", "diff"]:
        train_diffusion(*args, **kwargs)
