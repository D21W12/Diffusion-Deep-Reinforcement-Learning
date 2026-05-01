import random

import torch

from torch.nn import SmoothL1Loss
from torch.optim import AdamW

from .base_agent import Agent
from .experience_replay import ReplayMemory
from ..nn import DQNRedKnight


class DQNAgent(Agent):

    def __init__(
            self,
            n_actions: int,
            train: bool,
            lr: float = 2.5e-4,
            replay_size: int = 1000000,
            discount: float = 0.99,
            update_frequency: int = 4,
            target_update_frequency: int = 10000,
            batch_size: int = 32,
            replay_start_size: int = 50000,
            final_exploration_frame: int = 1000000,
            device: str = "cpu"
    ) -> None:
        """Constructor of the DQNAgent class."""

        super().__init__(n_actions)

        print(lr)

        self._device = device

        self._train = train

        self._discount = discount
        self._update_frequency = update_frequency
        self._target_update_frequency = target_update_frequency
        self._replay_start_size = replay_start_size
        self._final_exploration_frame = final_exploration_frame
        self._batch_size = batch_size

        self._dqn = DQNRedKnight(n_actions=n_actions).to(self._device)
        self._memory = ReplayMemory(
            N=replay_size,
        ) if train else None

        # Load same initial weights as dqn
        self._target_dqn = DQNRedKnight(n_actions=n_actions).to(self._device)
        self._target_dqn.load_state_dict(self._dqn.state_dict())

        self._optimizer = AdamW(
            params=self._dqn.parameters(),
            lr=lr,
        )
        self._criterion = SmoothL1Loss()

        self._epsilon = 1
        self._steps = 0

    def select_action(self, s) -> int:

        # With chance e select a random action (during training)
        if self._random_action():
            return torch.randint(self._n_actions, (1,)).item()

        s = s.to(device=self._device)
        with torch.no_grad():
            q_values = self._dqn(s.unsqueeze(0))
        return q_values.argmax().item()

    def observe(self, s, a, r, s_prime, t) -> None:

        if not self._train:
            raise Exception("Agent must be set to train mode before being able to observe experiences.")

        self._steps += 1

        self._memory.add(
            s=s,
            a=a,
            r=r,
            s_prime=s_prime,
            t=t
        )

        self._update_epsilon()

    def update(self) -> None:

        # Check whether agent is in train mode
        if not self._train:
            raise Exception("Agent must be set to train mode before being able to update checkpoints.")

        if self._is_update_step():
            self._update_network()

        if self._is_target_update_step():
            self._update_target_dqn()

    @property
    def _learning_steps(self):
        return max(0, self._steps - self._replay_start_size)

    @property
    def _updates(self):
        return (self._learning_steps - 1) // self._update_frequency

    def _update_network(self):

        s, a, r, s_prime, t = self._memory.sample(n=self._batch_size)

        # Move all tensors to the correct device
        s = s.to(self._device)
        a = a.to(self._device)
        r = r.to(self._device)
        s_prime = s_prime.to(self._device)
        t = t.to(self._device)

        self._optimizer.zero_grad()
        loss = self._loss(s, a, r, s_prime, t)
        loss.backward()  # Computing gradients
        self._optimizer.step()

    def _loss(self, s, a, r, s_prime, t) -> torch.Tensor:

        # We detach the computational graph of the target, because
        # we do not need to compute gradients for the target network.
        with torch.no_grad():
            targets = r + (self._discount * self._target_dqn(s_prime).max(dim=1).values) * (~t)
        q_values = self._dqn(s)
        q_values = q_values.gather(dim=1, index=a.unsqueeze(1)).squeeze(1)
        return self._criterion(q_values, targets)

    def _update_target_dqn(self) -> None:
        """
        Method for updating the target network, with the checkpoints
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
        if self._steps <= self._final_exploration_frame:
            self._epsilon -= 0.9 / self._final_exploration_frame
        elif self._learning_steps > self._final_exploration_frame:
            self._epsilon = 0.1

    def _random_action(self):
        if self._train:
            return (random.random() < self._epsilon) or self._is_starting_step()
        return random.random() < 0.05

    def _is_starting_step(self):
        return self._steps < self._replay_start_size

    def _is_update_step(self):
        if (self._steps - 1) < self._replay_start_size:
            return False
        return (self._steps - 1) % self._update_frequency == 0

    def _is_target_update_step(self):
        return (self._steps - 1) % self._target_update_frequency == 0

    def save(self, f):
        torch.save({
            'epsilon': self._epsilon,
            'steps': self._steps,
            'dqn_state_dict': self._dqn.state_dict(),
            'target_dqn_state_dict': self._target_dqn.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
        }, f)

    def save_memory(self, f):
        self._memory.save(f)

    def load(self, f) -> 'DQNAgent':
        data = torch.load(f, weights_only=False, map_location=self._device)

        self._epsilon = data['epsilon']
        self._steps = data['steps']

        self._dqn.load_state_dict(data['dqn_state_dict'])
        self._target_dqn.load_state_dict(data['target_dqn_state_dict'])

        self._optimizer.load_state_dict(data['optimizer_state_dict'])

        return self

    def load_memory(self, f):
        if not self._train:
            raise Exception("Agent must be set to train mode before being able to observe experiences.")
        self._memory.load(f)

    @staticmethod
    def _check_device(self, device: str) -> 'DQNAgent':
        OPTIONS = ["cuda", "mps", "cpu"]

        # Check if a correct/available device is given.
        if device not in OPTIONS:
            raise ValueError(f"Device should be one of {', '.join(OPTIONS)} not {device}!")
        elif device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Cuda not available!")
        elif device == "mps" and not torch.mps.is_available():
            raise ValueError(f"MPS not available!")
