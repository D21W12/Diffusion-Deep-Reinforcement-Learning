import gymnasium as gym

from abc import abstractmethod, ABC

import torch
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

        self._seed = None

    def set_seed(self, seed):
        self._seed = seed
        
        torch.seed(seed)



    @abstractmethod
    def run(self, *args, **kwargs):
        pass
