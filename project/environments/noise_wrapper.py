import gymnasium as gym
import torch
from torchvision.transforms.v2 import GaussianNoise


class NoiseWrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            sigma: float,
    ) -> None:
        super().__init__(env)
        self._sigma = sigma

    def _transform(self, s: torch.Tensor) -> torch.Tensor:
        s = (s - 0.5) * 2
        e = self._sigma * torch.randn_like(s)
        s = ((s + e) + 1) / 2
        return s

    def step(self, a: int):
        s, r, e, t, i = super().step(a)
        return (
            self._transform(s),
            r,
            e,
            t,
            i
        )
