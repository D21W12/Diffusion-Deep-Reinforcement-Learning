import gymnasium as gym

from abc import abstractmethod, ABC

from tqdm import tqdm

from ...agents import Agent


class Loop(ABC):

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ) -> None:

        self._env = env
        self._agent = agent

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
