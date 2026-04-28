import pandas as pd


class Evaluator:

    def __init__(self):
        self._episode_length = []
        self._cum_reward = []
        self._avg_reward = []

    def record(
            self,
            episode_length: int,
            cum_reward: float,
            avg_reward: float,
    ) -> None:
        self._episode_length.append(episode_length)
        self._cum_reward.append(cum_reward)
        self._avg_reward.append(avg_reward)

    def to_df(self) -> pd.DataFrame:
        data = {
            "episode length": self._episode_length,
            "cumulative reward": self._cum_reward,
            "average reward": self._avg_reward,
        }
        return pd.DataFrame(data)

    def to_csv(self, path: str) -> None:
        self.to_df().to_csv(path)
