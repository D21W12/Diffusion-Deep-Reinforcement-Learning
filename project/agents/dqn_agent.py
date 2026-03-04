from .experience_replay import ReplayMemory


class DQNAgent:

    def __init__(
            self
    ) -> None:
        """Constructor of the DQNAgent class."""

        super().__init__()

        self._train = False

        self._dqn = ...
        self._memory = ReplayMemory()
        self._epsilon = ...

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
