import pandas as pd


class Evaluator:

    def __init__(self):
        self._length = []
        self._reward = []

    def record(
            self,
            length: int,
            reward: float,
    ) -> None:
        self._length.append(length)
        self._reward.append(reward)

    def to_df(self) -> pd.DataFrame:
        data = {
            "length": self._length,
            "reward": self._reward,
        }
        return pd.DataFrame(data)

    def to_csv(self, path: str) -> None:
        self.to_df().to_csv(path)


class ExperimentEvaluator(Evaluator):

    def __init__(
        self,
        setup: str,
        environment: str,
    ):
        super().__init__()

        self._setup = setup
        self._environment = environment

        self._sigma_noise = []

    def record(
            self,
            length: int,
            reward: float,
            sigma_noise: float,
    ) -> None:
        super().record(length, reward)
        self._sigma_noise.append(sigma_noise)


    def to_df(self) -> pd.DataFrame:
        df = super().to_df()
        df["sigma_noise"] = self._sigma_noise
        df["setup"] = self._setup
        df["environment"] = self._environment
        return df
