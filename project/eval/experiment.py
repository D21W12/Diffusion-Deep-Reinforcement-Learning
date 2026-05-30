import argparse

from project.agents import DDQNAgent, DenoisingAgent, DiffusionAgent
from project.environments import BaseWrapper
from project.environments import NoiseWrapper
from project.environments.loops import EvaluationLoop
from project.eval.config import ExperimentConfig
from project.models import EDMMauMau, EDMSerie
from project.util.filters import MedianFilter
from project.util.evaluator import ExperimentEvaluator
from project.util.seed import set_seed

SETUPS = ["baseline", "median", "diffusion_naive", "diffusion_full"]

def experiment(
        setup: str,
        sigma_noise: float,
        output: str,
        dqn_checkpoint: str,
        diff_checkpoint: str,
        device: str = "cpu"
):

    config = ExperimentConfig()

    set_seed(config.seed)

    env = BaseWrapper.create_environment(config.environment)
    env = NoiseWrapper(env)

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
            **agent_kwargs
        )

    agent = agent.to(device)

    evaluator = ExperimentEvaluator(
        setup=setup,
        sigma_noise=sigma_noise
    )

    loop = EvaluationLoop(
        agent=agent,
        env=env,
        evaluator=evaluator
    )

    loop.run(config.episodes)

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

    parser.add_argument('-n', '--noise', type=float)

    parser.add_argument('-o', '--output', type=str)

    args = parser.parse_args()

    experiment(
        setup=args.setup,
        sigma_noise=args.noise,
        dqn_checkpoint=args.dqn,
        diff_checkpoint=args.diff,
        output=args.output
    )
    

if __name__ == "__main__":
    main()