import torch

from torch import Tensor


class ReplayMemory:

    def __init__(self, N: int):

        super().__init__()

        self._obs_shape = None
        self._memory_is_init = False

        self._N = N + 1  # Make up for the fact that the memory add the pointer can never be sampled.
        self._i = 0  # Pointer to the position where the next experience should be saved
        self._full = False

        # Experience buffers
        self._s = None
        self._a = None
        self._r = None
        # self._s_prime = None
        self._t = None

    def sample(self, n) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if not self._full and n > self._i:
            raise ValueError("Sample size can not be bigger than the buffer itself!")

        if self._empty():
            raise EmptyBufferError("Can not sample a buffer that does not contain any experiences!")

        # Computing the sample indices
        size = self._N if self._full else self._i

        # If the memory is full, we don't want to include the experience at i, since this if overwritten by an s_prime.
        if self._full:
            sample_idx = torch.randint(0, size - 1, size=(n,))
            sample_idx = (self._i + sample_idx + 1) % self._N
        else:  # Otherwise we can just sample everything
            sample_idx = torch.randint(0, size, size=(n,))

        return (
            self._s[sample_idx],
            self._a[sample_idx],
            self._r[sample_idx],
            # self._s_prime[sample_idx],
            self._s[sample_idx + 1 % self._N],
            self._t[sample_idx]
        )

    def add(self, s, a, r, s_prime, t) -> None:

        if not self._memory_is_init:
            self._init_memory(obs_shape=s.shape)

        # Save Experience
        self._s[self._i] = s
        self._a[self._i] = a
        self._r[self._i] = r
        # self._s_prime[self._i] = s_prime
        self._s[self._i + 1 % self._N] = s_prime
        self._t[self._i] = t

        self._increment_i()

    def _increment_i(self) -> None:

        if self._i == self._N - 1:
            self._i = 0
            self._full = True
            return
        self._i += 1

    def _empty(self) -> bool:
        return not self._full and self._i == 0

    def _init_memory(self, obs_shape: tuple | None = None, empty: bool = True) -> None:

        if empty:

            if obs_shape is None:
                raise ValueError("Observation shape needs to be specified when initializing an empty buffer!")

            self._s = torch.zeros(size=(self._N,) + obs_shape, dtype=torch.uint8)
            self._a = torch.zeros(size=(self._N,), dtype=torch.int)
            self._r = torch.zeros(size=(self._N,), dtype=torch.float)
            # self._s_prime = torch.zeros(size=(self._N,) + obs_shape, dtype=torch.uint8)
            self._t = torch.zeros(size=(self._N,), dtype=torch.bool)

        self._memory_is_init = True

    def state_dict(self) -> dict:

        if not self._memory_is_init:
            raise EmptyBufferError("Can't save uninitialized memory!")

        return {
            "N": self._N,
            "i": self._i,
            "full": self._full,
            "s": self._s,
            "a": self._a,
            "r": self._r,
            # "s_prime": self._s_prime,
            "t": self._t
        }

    def load_state_dict(self, state_dict: dict) -> None:

        self._N = state_dict["N"]
        self._i= state_dict["i"]
        self._full = state_dict["full"]
        self._s = state_dict["s"]
        self._a = state_dict["a"]
        self._r = state_dict["r"]
        # self._s_prime = state_dict["s_prime"]
        self._t = state_dict["t"]

        self._init_memory(empty=False)

    def save(self, f):
        torch.save(self.state_dict(), f)

    def load(self, f):
        self.load_state_dict(torch.load(f))


class EmptyBufferError(Exception):
    pass


if __name__ == "__main__":

    memory = ReplayMemory(
        N=5,
    )

    s = torch.randint(255, (1,))
    for i in range(10):
        s_prime = torch.randint(255, (1,))
        r = torch.randint(10, (1,))
        a = torch.randint(6, (1,))
        t = i % 2 == 0
        memory.add(s, a, r, s_prime, t)
        s = s_prime

    print(memory.sample(2))
