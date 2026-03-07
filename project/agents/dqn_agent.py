import torch

from torch.optim import SGD

from .experience_replay import ReplayMemory
from ..nn import DQNMilkyWay


class DQNAgent:
    # TODO: implement target network update frequency

    def __init__(
            self,
            lr: float,
            n_actions: int,
            obs_shape: tuple,
            replay_size: int,
            discount: float,
    ) -> None:
        """Constructor of the DQNAgent class."""

        super().__init__()

        self._train = False

        self._n_actions = n_actions
        self._obs_shape = obs_shape

        self._discount = discount

        self._dqn = DQNMilkyWay(n_actions=n_actions)
        self._target_dqn = DQNMilkyWay(n_actions=n_actions)
        print("Initializing replay")
        self._memory = ReplayMemory(
            N=replay_size,
            obs_shape=obs_shape
        )
        print("Done!")

        self._optimizer = SGD(
            params=self._dqn.parameters(),
            lr=lr
        )

        self._last_observation = None
        self._epsilon = 1


    def select_action(self, s) -> int:

        # Save the last observation
        self._last_observation = s

        # With chance e select a random action
        if torch.rand((1,)) < self._epsilon:
            return torch.randint(self._n_actions, (1,)).item()

        q_values = self._dqn(s.unsqueeze(0))
        return q_values.argmax().item()

    def observe(self, a, r, s_prime) -> None:
        self._memory.add(
            s=self._last_observation,
            a=a,
            r=r,
            s_prime=s_prime
        )

    def update(self, batch_size: int) -> None:

        # Check whether agent is in train mode
        if not self._train:
            raise Exception("Agent must be set to train mode before being able to update parameters.")

        s, a, r, s_prime = self._memory.sample(n=batch_size)

        loss = self._loss(s, a, r, s_prime)
        loss.backward()  # Computing gradients
        self._optimizer.step()

        self._update_target_dqn()
        self._update_epsilon()

    def _loss(self, s, a, r, s_prime) -> torch.Tensor:

        targets = r + self._discount * self._target_dqn(s_prime).amax(dim=1)

        # We detach the computational graph of the target, because we do not need to compute gradients
        # for the target network.
        targets = targets.detach()

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
