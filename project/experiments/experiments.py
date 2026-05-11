from project.agents import DQNAgent, Agent
from project.agents.diffusion_agent import DiffusionAgent
from project.environments import BaseWrapper, NoiseWrapper
from project.environments.loops import EvaluationLoop
from project.util.evaluator import Evaluator


class Experiment:
    AGENTS = ["DQN", "Diffusion"]

    def __init__(
            self,
            output: str,
            device: str,
    ):
        self._output = output
        self._device = device

        self._env = self.query_environment()
        self._agent = self.query_agent().to(device)

    def query_environment(self):
        id = input(f"Which environment do you want to test in?")
        return BaseWrapper.create_environment(id)

    def query_checkpoint(self, model: str = "DQN") -> str:
        f = input(f"Please enter path to checkpoint ({model}):\n")

    def query_agent(self) -> Agent:

        dqn_checkpoint = self.query_checkpoint()

        agent = input(f"Which agent do you want to test? ({','.join(self.AGENTS)}\n")
        kwargs = {"n_actions": self._env.action_space.n, "train": False}

        if agent.lower() == "dqn":
            print("Chose: dqn")
            return DQNAgent(**kwargs).load(dqn_checkpoint)
        elif agent.lower() == "diffusion":
            print("Chose: diffusion")
            agent = DiffusionAgent(**kwargs)
            diffusion_checkpoint = self.query_checkpoint("Diffusion")
            return agent.load(dqn_checkpoint, diffusion_checkpoint)
        else:
            raise ValueError(f"{agent} is not a valid choice!")

    def run(self):
        raise NotImplemented

    def save(self):
        raise NotImplemented


class Baseline(Experiment):
    EPISODES = 1_000

    def run(self):

        evaluator = Evaluator()
        loop = EvaluationLoop(
            agent=self._agent,
            env=self._env,
            evaluator=evaluator
        )
        loop.run(self.EPISODES)
        evaluator.to_csv(self._output)


class GaussianNoise(Experiment):

    def query_environment(self) -> float:
        env = super().query_environment()
        sigma = float(input(f"Give the standard deviation of the added noise:\n"))
        return NoiseWrapper(env, sigma)
