import argparse

import ale_py
import gymnasium as gym

from project.agents import DQNAgent
from project.environments import BaseWrapper, NoiseWrapper
from project.environments.loops import EvaluationLoop, TestingLoop
from project.eval.config import DQNEvalConfig
from project.util.evaluator import Evaluator

def manual_eval(
        checkpoint_path: str,
        device: str,
) -> None:
    gym.register_envs(ale_py)

    config = DQNEvalConfig()

    print(f"Using: {device}")

    env = BaseWrapper.create_environment(config.environment, render_mode="human")

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


def empirical_eval(
        checkpoint_path: str,
        output_path: str,
        device: str,
        number: int,
) -> None:
    gym.register_envs(ale_py)

    config = DQNEvalConfig()

    print(f"Using: {device}")

    env = BaseWrapper.create_environment(config.environment)

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


def main():

    parser = argparse.ArgumentParser(
        prog='Project eval',
        description='This program manages eval for my BSc Thesis project.',
        epilog='That are all commands >.<'
    )

    parser.add_argument('-c', '--checkpoint', required=True)

    parser.add_argument('-d', '--device', default='cpu')

    parser.add_argument("--manual", action='store_true')

    parser.add_argument('-n', '--number', type=int)
    parser.add_argument('-o', '--output')

    args = parser.parse_args()

    if args.manual:
        manual_eval(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    else:
        empirical_eval(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            device=args.device,
            number=args.number
        )


if __name__ == "__main__":
    main()
