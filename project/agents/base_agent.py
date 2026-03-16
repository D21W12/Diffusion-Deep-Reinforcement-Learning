from abc import ABC


class Agent(ABC):

    def __init__(self, n_actions: int) -> None:
        self._n_actions = n_actions

    def select_action(self, s) -> int:
        pass

    def observe(self, s, a, r, s_prime, t):
        pass

    def update(self):
        pass