from abc import abstractmethod, ABC

from .on_device import OnDevice


class Agent(OnDevice, ABC):

    def __init__(
            self,
            train: bool
    ):
        self._train = train

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
