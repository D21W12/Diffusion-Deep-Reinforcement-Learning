import torch

from torch import Tensor


class ReplayMemory:

    def __init__(self, N: int, obs_shape: tuple):

        self._N = N
        self._i = 0  # Pointer to the last saved experience
        self._full = False

        # Experience buffers
        self._s = torch.zeros(size=(N,) + obs_shape)
        self._a = torch.zeros(size=(N,))
        self._r = torch.zeros(size=(N,))
        self._s_prime = torch.zeros(size=(N,)  + obs_shape)

    def sample(self, n) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        size = self._N if self._full else self._i
        sample_idx = torch.randint(size, size=(n,))

        return (
            self._s[sample_idx],
            self._a[sample_idx],
            self._r[sample_idx],
            self._s_prime[sample_idx]
        )

    def add(self, s, a, r, s_prime) -> None:

        # Save Experience
        self._s[self._i] = s
        self._a[self._i] = a
        self._r[self._i] = r
        self._s_prime[self._i] = s_prime

        self._increment_i()

    def _increment_i(self) -> None:


        if self._i == self._N:
            self._i = 0
            self._full = True
            return
        self._i += 1
