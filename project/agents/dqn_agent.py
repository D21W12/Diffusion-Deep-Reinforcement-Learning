import torch

from typing import override
from torch.optim import SGD

from .base_agent import Agent
from .experience_replay import ReplayMemory
from ..nn import DQNMilkyWay


class DQNAgent(Agent):
    # TODO: Remove being able to change train mode

    def __init__(
            self,
            n_actions: int,
            obs_shape: tuple,
            train: bool,
            lr: float = 25e-4,
            replay_size: int = 1000000,
            discount: float = 0.99,
            update_frequency: int = 4,
            target_update_frequency: int = 4,
            batch_size: int = 32,
            frame_skips: int = 4,
            replay_start_size: int = 50000,
            final_exploration_frame: int = 1000000
    ) -> None:
        """Constructor of the DQNAgent class."""

        super().__init__(train=train, n_actions=n_actions)

        self._device = "cpu"

        self._obs_shape = obs_shape

        self._discount = discount
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency
        self._frame_skips = frame_skips
        self._replay_start_size = replay_start_size
        self._final_exploration_frame = final_exploration_frame
        self._batch_size = batch_size

        self._dqn = DQNMilkyWay(n_actions=n_actions).to(self._device)
        self._target_dqn = DQNMilkyWay(n_actions=n_actions).to(self._device)
        self._memory = ReplayMemory(
            N=replay_size,
            obs_shape=obs_shape
        ).to(self._device) if train else None

        self._optimizer = SGD(
            params=self._dqn.parameters(),
            lr=lr
        )

        self._last_observation = None
        self._last_action = None
        self._epsilon = 1
        self._frames_seen = 0

    def select_action(self, s) -> int:

        self._frames_seen += 1

        # We only select a new action on non-skipped frames
        if self._is_action_select_step():
            self._last_action = self._select_action(s)

        return self._last_action

    def _select_action(self, s) -> int:

        # With chance e select a random action (during training)
        if self._random_action():
            return torch.randint(self._n_actions, (1,)).item()

        s = s.to(self._device)
        with torch.no_grad():
            q_values = self._dqn(s.unsqueeze(0))
        return q_values.argmax().item()

    def _random_action(self):
        return self._train and (torch.rand((1,)) < self._epsilon or self._is_starting_step())

    def _is_action_select_step(self):
        return (self._frames_seen - 1) % self._frame_skips == 0

    def _is_starting_step(self):
        return self._frames_seen < self._replay_start_size

    def observe(self, s, a, r, s_prime) -> None:

        if not self._train:
            raise Exception("Agent must be set to train mode before being able to observe experiences.")

        # Storing the last observation
        self._last_observation = s

        self._memory.add(
            s=s,
            a=a,
            r=r,
            s_prime=s_prime
        )

    def update(self) -> None:

        # Check whether agent is in train mode
        if not self._train:
            raise Exception("Agent must be set to train mode before being able to update parameters.")

        if self._is_update_step():
            self._update_network()

        self._update_epsilon()

    def _update_network(self):

        s, a, r, s_prime = self._memory.sample(n=self._batch_size)

        device = self._device
        s, r, s_prime = s.to(device), r.to(device), s_prime.to(device)

        loss = self._loss(s, a, r, s_prime)
        loss.backward()  # Computing gradients
        self._optimizer.step()
        self._optimizer.zero_grad()

        if self._is_target_update_step():
            self._update_target_dqn()

    def _loss(self, s, a, r, s_prime) -> torch.Tensor:

        # We detach the computational graph of the target, because we do not need to compute gradients
        # for the target network.
        with torch.no_grad():
            targets = r + self._discount * self._target_dqn(s_prime).amax(dim=1)
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
        if self._frames_seen < self._final_exploration_frame:
            self._epsilon -= 0.9 / self._final_exploration_frame

    def _is_update_step(self):
        if (self._frames_seen - 1) < self._replay_start_size:
            return False
        return (self._learning_frames - 1) % (self._update_frequency * self._frame_skips) == 0

    def _is_target_update_step(self):
        frequency = self._target_update_frequency * self._update_frequency * self._frame_skips
        return (self._learning_frames - 1) % frequency == 0

    @property
    def _learning_frames(self):
        return self._frames_seen - self._replay_start_size

    def save(self, f):
        torch.save({
            'last_observation': self._last_observation,
            'last_action': self._last_action,
            'epsilon': self._epsilon,
            'frames_seen': self._frames_seen,
            'dqn_state_dict': self._dqn.state_dict(),
            'target_dqn_state_dict': self._dqn.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict()
        }, f)

    def load(self, f):
        data = torch.load(f, weights_only=True)

        self._last_observation = data['last_observation']
        self._last_action = data['last_action']
        self._epsilon = data['epsilon']
        self._frames_seen = data['frames_seen']

        self._dqn.load_state_dict(data['dqn_state_dict'])
        self._target_dqn.load_state_dict(data['target_dqn_state_dict'])

    @override
    def to(self, device: str) -> 'DQNAgent':
        super().to(device)

        self._dqn.to(device)
        self._target_dqn.to(device)
        self._memory.to(device)

        return self
