import gymnasium as gym
import torch
from torchvision import transforms
from project.util.transforms import Difference


class NoiseWrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            sigma: float,
    ) -> None:
        super().__init__(env)
        self._sigma = sigma

    def _transform_observation(self, s: torch.Tensor) -> torch.Tensor:
        e = torch.randn_like(s) * self._sigma
        return s + e * 0.5  # Scale by 0.5 to account for range [0, 1] instead of [-1, 1]

    def step(self, a: int):
        s, r, e, t, i = super().step(a)
        return (
            self._transform_observation(s),
            r,
            e,
            t,
            i
        )
    
    def reset(self, *args, **kwargs):
        s, i = super().reset(*args, **kwargs)
        return self._transform_observation(s), i

