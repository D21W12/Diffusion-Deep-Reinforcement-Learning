from abc import abstractmethod

import gymnasium as gym

from tqdm import tqdm

from ...agents import Agent


class Loop:

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ) -> None:

        self._env = env
        self._agent = agent

    @abstractmethod
    def run(self, frames: int):
        pass
