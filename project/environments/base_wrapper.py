import numpy as np
import torch
import gymnasium as gym
from torch import Tensor


class BaseWrapper(gym.Wrapper):
    """
    Base
    """
    FIRE = 1

    def __init__(
            self,
            env: gym.Env,
    ) -> None:
        """
        Constructor of BaseEnvWrapper

        :param env: the environment to wrap
        :param stack_frames: the number of frames to stack
        """

        env = self._wrap_env(
            env=env,
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
        return torch.from_numpy(s).to(torch.uint8)

    def _wrap_env(
            self,
            env: gym.Env,
    ) -> gym.Env:
        env = gym.wrappers.MaxAndSkipObservation(env, skip=4)
        env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env = gym.wrappers.ClipReward(env, min_reward=-1, max_reward=1)

        return env

    @classmethod
    def create_environment(cls, id, render_mode: str | None = None):
        env = gym.make(id, render_mode=render_mode, frameskip=1)
        env = cls(env)
        return env
