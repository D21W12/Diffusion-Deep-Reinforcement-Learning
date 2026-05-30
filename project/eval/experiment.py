from project.agents import DDQNAgent, DenoisingAgent, DiffusionAgent
from project.environments import BaseWrapper
from project.environments import NoiseWrapper
from project.environments.loops import EvaluationLoop
from project.eval.config import ExperimentConfig
from project.models import EDMMauMau, EDMSerie
from project.util.denoisers import MedianFilter
from project.util.evaluator import ExperimentEvaluator

SETUPS = ["baseline1", "median", "diffusion_naive", "diffusion_full"]

def experiment(
        setup: str,
        sigma_noise: float,
        output: str,
        dqn_checkpoint: str,
        diff_checkpoint: str,
        device: str = "cpu"
):
    config = ExperimentConfig()

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
    elif setup == "baseline1":
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




    
