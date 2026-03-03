class Agent:

    def __init__(self) -> None:
        self._train = False
        pass

    def train(self) -> bool:
        self._train = True
        return self._train

    def test(self) -> bool:
        self._train = False
        return self._train
