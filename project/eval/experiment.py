import argparse

import ale_py
import gymnasium as gym

from project.agents import DDQNAgent, DenoisingAgent, DiffusionAgent
from project.environments import BaseWrapper
from project.environments import NoiseWrapper
from project.environments.loops import ExperimentLoop
from project.eval.config import ExperimentConfig
from project.models import EDMMauMau, EDMSerie
from project.util.filters import MedianFilter
from project.util.evaluator import ExperimentEvaluator
from project.util.seed import set_seed

SETUPS = ["baseline", "median", "diffusion_naive", "diffusion_full"]

def experiment(
        config: ExperimentConfig,
        setup: str,
        sigma_noise: float,
        dqn_checkpoint: str,
        diff_checkpoint: str | None,
        evaluator: ExperimentEvaluator,
        device: str = "cpu"
):

    set_seed(config.seed)

    gym.register_envs(ale_py)

    env = BaseWrapper.create_environment(config.environment)
    env = NoiseWrapper(env, sigma=sigma_noise)

    agent_kwargs = {
        "train": False, 
        "epsilon": config.epsilon, 
        "n_actions": env.action_space.n
        }

    agent = None

    if setup not in SETUPS:
        raise ValueError(f"The given experimental '{setup}' setup does not exist.")
    elif setup == "baseline":
        agent = DDQNAgent(**agent_kwargs)
    elif setup == "median":
        median_filter = MedianFilter(kernel_size=config.kernel_size)
        agent = DenoisingAgent(
            denoiser=median_filter,
            **agent_kwargs
            )
    elif setup == "diffusion_naive":
        model = EDMMauMau.from_checkpoint(
            diff_checkpoint, 
            device=device,
            sigma_noise=sigma_noise,
            )
        agent = DiffusionAgent(
            model=model,
            sigma_noise=sigma_noise,
            **agent_kwargs
        )
    elif setup == "diffusion_full":
        model = EDMSerie.from_checkpoint(
            diff_checkpoint, 
            device=device,
            sigma_noise=sigma_noise,
            N=config.N
            )
        agent = DiffusionAgent(
            model=model,
            sigma_noise=sigma_noise,
            **agent_kwargs
        )

    agent.to(device)
    agent.load(dqn_checkpoint)

    loop = ExperimentLoop(
        agent=agent,
        env=env,
        evaluator=evaluator,
        sigma_noise=sigma_noise
    )

    loop.run(config.episodes)


def single_experiment(
        setup: str,
        sigma_noise: float,
        output: str,
        dqn_checkpoint: str,
        diff_checkpoint: str | None,
        device: str = "cpu"
        ):
    
    config = ExperimentConfig()
    evaluator = ExperimentEvaluator(
            setup=setup,
            environment=config.environment
        ) 

    experiment(
        config=config,
        setup=setup,
        sigma_noise=sigma_noise,
        dqn_checkpoint=dqn_checkpoint,
        diff_checkpoint=diff_checkpoint,
        evaluator=evaluator,
        device=device
    )

    evaluator.to_csv(output)


def multiple_experiment(
        setup: str,
        output: str,
        dqn_checkpoint: str,
        diff_checkpoint: str | None,
        device: str = "cpu"
        ):
    
    config = ExperimentConfig()
    evaluator = ExperimentEvaluator(
            setup=setup,
            environment=config.environment
        ) 
    
    for sigma_noise in config.noise_levels:
        experiment(
            config=config,
            setup=setup,
            sigma_noise=sigma_noise,
            dqn_checkpoint=dqn_checkpoint,
            diff_checkpoint=diff_checkpoint,
            evaluator=evaluator,
            device=device
        )

    evaluator.to_csv(output)


def main():

    parser = argparse.ArgumentParser(
        prog='Project experiment',
        description='This program manages running experiments for DiffRL.',
        epilog='That are all commands >.<'
    )

    parser.add_argument('--dqn', required=True)
    parser.add_argument('--diffusion', required=False, default=None)

    parser.add_argument('-d', '--device', default='cpu')

    parser.add_argument('-s', '--setup', type=str)

    parser.add_argument('-n', '--noise', default=None, type=float)
    parser.add_argument('-m', '--multiple', default=False, action='store_true')

    parser.add_argument('-o', '--output', type=str)

    args = parser.parse_args()

    if args.multiple:
        multiple_experiment(
            setup=args.setup,
            dqn_checkpoint=args.dqn,
            diff_checkpoint=args.diffusion,
            output=args.output,
            device=args.device,
        )
    elif args.noise is not None:
        single_experiment(
            setup=args.setup,
            sigma_noise=args.noise,
            dqn_checkpoint=args.dqn,
            diff_checkpoint=args.diffusion,
            output=args.output,
            device=args.device,
        )
    else:
        raise ValueError("Please pass a noise level or start mutliple experiments!")
    

if __name__ == "__main__":
    main()
