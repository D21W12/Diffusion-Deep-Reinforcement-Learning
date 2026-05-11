import numpy as np
import gymnasium as gym
import torch
from torch import Tensor
from torchvision.transforms.v2 import GaussianNoise

from project.environments import BaseWrapper


class NoiseWrapper(BaseWrapper):

    def __init__(
            self,
            env: gym.Env,
            sigma: float,
    ) -> None:
        super().__init__(env)

        self._sigma = sigma
        self._transform = GaussianNoise(sigma=sigma)

    def _transform_observation(self, s: np.ndarray) -> Tensor:
        s = super()._transform_observation(s)
        return self._transform(s)
