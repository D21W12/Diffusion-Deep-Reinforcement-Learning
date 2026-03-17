import torch

from torch import Tensor


class ReplayMemory:
    # TODO: Fix sampling to be without replacement.

    def __init__(self, N: int, obs_shape: tuple):

        super().__init__()

        self._N = N
        self._i = 0  # Pointer to the last saved experience
        self._full = False

        # Experience buffers
        self._s = torch.zeros(size=(N,) + obs_shape, dtype=torch.uint8)
        self._a = torch.zeros(size=(N,), dtype=torch.int)
        self._r = torch.zeros(size=(N,), dtype=torch.float)
        self._s_prime = torch.zeros(size=(N,) + obs_shape, dtype=torch.uint8)
        self._t = torch.zeros(size=(N,), dtype=torch.bool)

    def sample(self, n) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if not self._full and n > self._i:
            raise ValueError("Sample size can not be bigger than the buffer itself!")

        if self._empty():
            raise EmptyBufferError("Can not sample a buffer that does not contain any experiences!")

        size = self._N if self._full else self._i
        sample_idx = torch.randint(0, size, size=(n,))

        return (
            self._s[sample_idx],
            self._a[sample_idx],
            self._r[sample_idx],
            self._s_prime[sample_idx],
            self._t[sample_idx]
        )

    def add(self, s, a, r, s_prime, t) -> None:

        # Save Experience
        self._s[self._i] = s
        self._a[self._i] = a
        self._r[self._i] = r
        self._s_prime[self._i] = s_prime
        self._t[self._i] = t

        self._increment_i()

    def _increment_i(self) -> None:

        if self._i == self._N - 1:
            self._i = 0
            self._full = True
            return
        self._i += 1

    def _empty(self):
        return not self._full and self._i == 0


class EmptyBufferError(Exception):
    pass


if __name__ == "__main__":

    memory = ReplayMemory(
        N=100,
        obs_shape=(1,)
    )

    s = torch.randint(255, (1,))
    for i in range(10):
        s_prime = torch.randint(255, (1,))
        r = torch.randint(10, (1,))
        a = torch.randint(6, (1,))
        memory.add(s, a, r, s_prime)

    print(memory.sample(2))
