from abc import abstractmethod, ABC

from .on_device import OnDevice


class Agent(OnDevice, ABC):

    def __init__(
            self,
            n_actions: int,
            train: bool
    ):
        self._train = train
        self._n_actions = n_actions
