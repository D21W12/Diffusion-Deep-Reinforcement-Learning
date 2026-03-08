import torch

from torch.optim import SGD

from .experience_replay import ReplayMemory
from ..nn import DQNMilkyWay
from ..util import Timer


class DQNAgent:
    # TODO: implement target network update frequency

    def __init__(
            self,
            lr: float,
            n_actions: int,
            obs_shape: tuple,
            replay_size: int = 1000000,
            discount: float = 0.99,
            update_frequency: int = 4,
            target_update_frequency: int = 4,
            batch_size: int = 32,
    ) -> None:
        """Constructor of the DQNAgent class."""

        super().__init__()

        self._train = False

        self._n_actions = n_actions
        self._obs_shape = obs_shape

        self._discount = discount
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency
        self._batch_size = batch_size

        self._dqn = DQNMilkyWay(n_actions=n_actions)
        self._target_dqn = DQNMilkyWay(n_actions=n_actions)
        self._memory = ReplayMemory(
            N=replay_size,
            obs_shape=obs_shape
        )

        self._optimizer = SGD(
            params=self._dqn.parameters(),
            lr=lr
        )

        self._last_observation = None
        self._epsilon = 1
        self._updates = 0
        self._actions_selected = 0


    def select_action(self, s) -> int:

        # Save the last observation
        self._last_observation = s
        self._actions_selected += 1

        # With chance e select a random action (during training)
        if self._train and torch.rand((1,)) < self._epsilon:
            return torch.randint(self._n_actions, (1,)).item()

        with torch.no_grad():
            q_values = self._dqn(s.unsqueeze(0))
        return q_values.argmax().item()

    def observe(self, a, r, s_prime) -> None:
        self._memory.add(
            s=self._last_observation,
            a=a,
            r=r,
            s_prime=s_prime
        )

    def update(self) -> None:

        # Check whether agent is in train mode
        if not self._train:
            raise Exception("Agent must be set to train mode before being able to update parameters.")

        if self._is_gradient_step():
            self._update_network()

        self._update_epsilon()

    def _update_network(self):
        self._updates += 1

        s, a, r, s_prime = self._memory.sample(n=self._batch_size)

        loss = self._loss(s, a, r, s_prime)
        loss.backward()  # Computing gradients
        self._optimizer.step()
        self._optimizer.zero_grad()

        if self._is_target_update_step():
            self._update_target_dqn()

    def _loss(self, s, a, r, s_prime) -> torch.Tensor:

        targets = r + self._discount * self._target_dqn(s_prime).amax(dim=1)

        # We detach the computational graph of the target, because we do not need to compute gradients
        # for the target network.
        with torch.no_grad():
            targets = targets

        q_values = self._dqn(s).amax(dim=1)
        return ((targets - q_values)**2).mean()

    def _update_target_dqn(self) -> None:
        """
        Method for updating the target network, with the parameters
        of the non-target network.
        """
        self._target_dqn.load_state_dict(self._dqn.state_dict())

    def _update_epsilon(self) -> None:
        """
        Method for updating epsilon.

        Current strategy:
        - Annealed linearly from 1 to 0.1 over 1 million updates.
        - Clipped at 0.1
        """
        self._epsilon = min(0.1, self._epsilon - 0.9 * 1e-6)

    def _is_gradient_step(self):
        return self._actions_selected % self._update_frequency == 0

    def _is_target_update_step(self):
        return self._updates % self._target_update_frequency == 0

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
