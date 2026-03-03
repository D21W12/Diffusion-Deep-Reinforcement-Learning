class ReplayMemory[S]:

    def __init__(self):
        self._experiences: list[Experience[S]] = []

    def sample(self) -> 'Experience[S]':
        raise NotImplementedError

    def add(self, e: 'Experience[S]') -> None:
        self._experiences.append(e)


class Experience[S]:

    def __init__(
            self,
            s: S,
            a: int,
            r: float,
            s_prime: S,
    ):
        self._experience: tuple[S, int, float, S] =  (s, a, r, s_prime)

    @property
    def s(self) -> S:
        return self._experience[0]

    @property
    def a(self) -> int:
        return self._experience[1]

    @property
    def r(self) -> float:
        return self._experience[2]

    @property
    def s_prime(self) -> S:
        return self._experience[3]
