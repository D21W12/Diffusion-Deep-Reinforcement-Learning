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
        self._length.append(episode_length)
        self._reward.append(cum_reward)

    def to_df(self) -> pd.DataFrame:
        data = {
            "length": self._episode_length,
            "reward": self._reward,
        }
        return pd.DataFrame(data)

    def to_csv(self, path: str) -> None:
        self.to_df().to_csv(path)


class ExperimentEvaluator(Evaluation):

    def __init__(
        self,
        setup: str,
        sigma_noise: float,
    ):
        super().__init__()
        self._setup = setup
        self._sigma_noise = sigma_noise

    def to_df(self) -> pd.DataFrame:
        df = super().to_df()
        df["setup"] = self._setup
        df["sigma_noise"] = self._sigma_noise
