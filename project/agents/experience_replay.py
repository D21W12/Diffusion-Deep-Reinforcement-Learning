class ReplayMemory[ObsType]:

    def __init__(self, N: int):
        self._experiences: list[Experience[ObsType]] = []
        self._N = N

    def sample(self, n: int) -> 'Experience[ObsType]':
        raise NotImplementedError

    def add(self, e: 'Experience[ObsType]') -> None:
        self._experiences.append(e)

        # If the replay memory is full, remove the first experience
        if len(self._experiences) > self._N:
            self._experiences.pop(0)


class Experience[ObsType]:

    def __init__(
            self,
            s: ObsType,
            a: int,
            r: float,
            s_prime: ObsType,
    ):
        self._experience: tuple[ObsType, int, float, ObsType] =  (s, a, r, s_prime)

    @property
    def s(self) -> ObsType:
        return self._experience[0]

    @property
    def a(self) -> int:
        return self._experience[1]

    @property
    def r(self) -> float:
        return self._experience[2]

    @property
    def s_prime(self) -> ObsType:
        return self._experience[3]
