import ale_py
import gymnasium as gym
import torch

from torchvision.transforms import transforms

from project.agents import DQNAgent
from project.environments import BaseWrapper
from project.environments.loops import EvaluationLoop, TestingLoop
from project.evaluation.config import DiffEvalConfig
from project.models import EDMEvelynn
from project.util.evaluator import Evaluator

ENVIRONMENT = "ALE/Breakout-v5"


def manual_eval_dqn(
        checkpoint_path: str,
        device: str,
) -> None:
    gym.register_envs(ale_py)

    print(f"Using: {device}")

    env = BaseWrapper.create_environment(ENVIRONMENT, render_mode="human")

    agent = DQNAgent(
        train=False,
        n_actions=env.action_space.n,
        device=device,
    ).load(checkpoint_path)

    loop = TestingLoop(
        env=env,
        agent=agent,
    )
    loop.run()


def empirical_eval_dqn(
        checkpoint_path: str,
        output_path: str,
        device: str,
        number: int,
) -> None:
    gym.register_envs(ale_py)

    print(f"Using: {device}")

    env = BaseWrapper.create_environment(ENVIRONMENT)

    agent = DQNAgent(
        train=False,
        n_actions=env.action_space.n,
        device=device,
    ).load(checkpoint_path)

    evaluator = Evaluator()

    loop = EvaluationLoop(
        env=env,
        agent=agent,
        evaluator=evaluator
    )
    loop.run(number)

    evaluator.to_csv(output_path)


def manual_eval_diff(
        checkpoint_path: str,
        device: str,
        network: str,
        *args,
        **kwargs,
) -> None:

    config = DiffEvalConfig(
        checkpoint_path=checkpoint_path,
        device=device,
        network=network,
        *args,
        **kwargs
    )

    model = EDMEvelynn(
        img_resolution=config.resolution,
        img_channels=config.in_channels,
        start_channels=config.start_channels,
        channel_mult=config.channel_multipliers,
        num_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
        network=config.network
    ).to(config.device)

    print("Loading checkpoint...")
    model.load(config.checkpoint_path)
    print("Loaded checkpoint!")

    # TODO: finish


def evaluate(
        model: str,
        manual: bool,
        *args,
        **kwargs,
) -> None:
    if model in ["dqn", "rl"] and manual:
        manual_eval_dqn(*args, **kwargs)
    elif model in ["dqn", "rl"]:
        empirical_eval_dqn(*args, **kwargs)
    elif model in ["diffusion", "diff"] and manual:
        manual_eval_diff(*args, **kwargs)
