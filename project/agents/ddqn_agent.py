import torch

from .dqn_agent import DQNAgent


class DDQNAgent(DQNAgent):

    def _loss(self, s, a, r, s_prime, t) -> torch.Tensor:
        # We detach the computational graph of the target, because
        # we do not need to compute gradients for the target network.
        with torch.no_grad():
            a_prime = self._dqn(s_prime).max(dim=1).indices
            q_s_prime = self._target_dqn(s_prime).gather(dim=1, index=a_prime.unsqueeze(1)).squeeze(1)
            targets = r + (self._discount * q_s_prime) * (~t)

        q_values = self._dqn(s)
        q_values = q_values.gather(dim=1, index=a.unsqueeze(1)).squeeze(1)
        return self._criterion(q_values, targets)
