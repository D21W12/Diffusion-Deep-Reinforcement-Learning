import torch

from .experience_replay import ReplayMemory
from ..nn import DQNMilkyWay


class DQNAgent:

    def __init__(
            self,
            n_actions: int,
            replay_size: int,
            discount: float,
            epsilon: float,
    ) -> None:
        """Constructor of the DQNAgent class."""

        super().__init__()

        self._train = False

        self._n_actions = n_actions

        self._dqn = DQNMilkyWay(n_actions=n_actions)
        self._target_dqn = DQNMilkyWay(n_actions=n_actions)
        self._memory = ReplayMemory(N=replay_size)

        self._epsilon = epsilon
        self._discount = discount

    def select_action(self, s) -> int:

        # With chance e select a random action
        if torch.rand() < self._epsilon:
            return torch.randint(self._n_actions, (1,)).item()

        return self._dqn(s).argmax().item()

    def observe(self, s):
        pass

    def update(self):

        # Check whether agent is in train mode
        if not self._train:
            raise Exception("Agent must be set to train mode before being able to update parameters.")

        s, a, r, s_prime = self._memory.sample()
        loss = self._loss(s, a, r, s_prime)
        loss.backward()  # Computing gradients

    def _loss(self, s, a, r, s_prime) -> torch.Tensor:
        targets = r + self._discount * self._target_dqn(s_prime).max(dim=0)
        q_values = self._dqn(s)[a]
        return (targets - q_values)**2

    def train(self) -> bool:
        """
        Method for setting the agent in training mode. Inheritance implementations
        should set all training settings using this method.
        """
        self._train = True
        return self._train

    def test(self) -> bool:
        """
        Method for setting the agent in testing mode. Inheritance implementations
        should set all test settings using this method.
        """
        self._train = False
        return self._train
