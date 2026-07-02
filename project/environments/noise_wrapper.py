import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from project.util.transforms import Difference


class NoiseWrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env,
            sigma: float,
            save_history: bool = False,
    ) -> None:
        super().__init__(env)
        self._sigma = sigma
        if save_history:
            self._history = []
            self._history_noisy = []
        else:
            self._history = None
            self._history_noisy = None

    def _transform_observation(self, s: torch.Tensor) -> torch.Tensor:
        e = torch.randn_like(s) * self._sigma
        o = s + e * 0.5  # Scale by 0.5 to account for range [0, 1] instead of [-1, 1]
        self._history.append(s)
        self._history_noisy.append(o)
        return o

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
    
    def save_history(self, f, duration: int = 20):
        clean = [Image.fromarray((image.clip(0, 1) * 510).clip(0, 255).mean(0).to(torch.uint8).numpy()) for image in self._history]
        noisy = [Image.fromarray((image.clip(0, 1) * 510).clip(0, 255).mean(0).to(torch.uint8).numpy()) for image in self._history_noisy]
        noisy[0].save(f"{f}_noisy.gif", save_all=True, append_images=noisy[1:], duration=duration, optimize=False)
        clean[0].save(f"{f}.gif", save_all=True, append_images=clean[1:], duration=duration, optimize=False)


