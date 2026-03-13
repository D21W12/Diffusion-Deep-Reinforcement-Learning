import random

from .base_agent import Agent


class RandomAgent(Agent):

    def select_action(self, s):
        return random.randint(0, self._n_actions - 1)
