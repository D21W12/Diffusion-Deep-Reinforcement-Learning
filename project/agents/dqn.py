from . import Agent
from .replay_memory import ReplayMemory


class DQNAgent(Agent):

    def __init__(
            self
    ) -> None:
        super().__init__()

        self._dqn = ...
        self._memory = ReplayMemory()
