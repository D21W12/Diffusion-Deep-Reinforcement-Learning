import gymnasium as gym
from torchvision.transforms.v2 import GaussianNoise


class NoiseWrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            sigma: float,
    ) -> None:
        super().__init__(env)
        self._transform = GaussianNoise(sigma=sigma)

    def step(self, a: int):
        s, r, e, t, i = super().step(a)
        return (
            self._transform(s),
            r,
            e,
            t,
            i
        )
