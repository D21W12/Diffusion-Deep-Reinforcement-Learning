import numpy as np
import torch
import gymnasium as gym
from torch import Tensor


class BaseWrapper(gym.Wrapper):
    """
    Base
    """

    def __init__(
            self,
            env: gym.Env,
            frame_stack_size: int = 4,
            grey_scale: bool = True,
    ) -> None:
        """
        Constructor of BaseEnvWrapper

        :param env: the environment to wrap
        :param stack_frames: the number of frames to stack
        """

        env = self._wrap_env(
            env=env,
            frame_stack_size=frame_stack_size,
            grey_scale=grey_scale,
        )
        super().__init__(env)

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

    def _transform_observation(self, s: np.ndarray) -> Tensor:
        return torch.from_numpy(s).to(torch.float32)

    def _wrap_env(
            self,
            env: gym.Env,
            frame_stack_size: int,
            grey_scale: bool,
    ) -> gym.Env:

        def _transform(s: np.ndarray) -> Tensor:
            return torch.from_numpy(s).to(torch.float32)

        if grey_scale: env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, frame_stack_size)
        env = gym.wrappers.ClipReward(env, -1, 1)
        return env
