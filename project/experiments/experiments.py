from project.agents import DQNAgent, Agent
from project.agents.diffusion_agent import DiffusionAgent
from project.environments import BaseWrapper


class Experiment:
    AGENTS = ["DQN", "Diffusion"]

    def __init__(
            self,
            output: str,
            device: str,
            checkpoints: dict,
    ):
        self._output = output
        self._device = device

        self._env = self.query_environment()
        self._agent = self.query_agent().to(device).load(**checkpoints)

    def query_environment(self):
        id = input(f"Which environment do you want to test in?")
        return BaseWrapper.create_environment(id)

    def query_agent(self) -> Agent:
        agent = input(f"Which agent do you want to test? ({','.join(self.AGENTS)}")
        kwargs = {"n_actions": self._env.action_space.n, "train": False}
        if agent.lower() == "dqn":
            print("Chose: dqn")
            return DQNAgent(**kwargs)
        elif agent.lower() == "diffusion":
            print("Chose: diffusion")
            return DiffusionAgent(**kwargs)
        else:
            raise ValueError(f"{agent} is not a valid choice!")

    def run(self):
        raise NotImplemented

    def save(self):
        raise NotImplemented
