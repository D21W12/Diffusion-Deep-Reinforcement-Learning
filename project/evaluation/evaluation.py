import ale_py
import gymnasium as gym
import torch

from project.agents import DQNAgent
from project.environments import BaseWrapper
from project.environments.loops import EvaluationLoop, TestingLoop
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
        episodes: int,
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
    loop.run(episodes)

    evaluator.to_csv(output_path)


def evaluate(
        manual: bool,
        *args,
        **kwargs,
) -> None:
    if manual:
        manual_eval_dqn(*args, **kwargs)
    else:
        empirical_eval_dqn(*args, **kwargs)
