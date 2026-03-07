import torch
import gymnasium as gym


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

    @staticmethod
    def _wrap_env(
            env: gym.Env,
            frame_stack_size: int,
            grey_scale: bool,
    ) -> gym.Env:

        if grey_scale: env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, frame_stack_size)
        env = gym.wrappers.NumpyToTorch(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda s: s.to(torch.float32),
            env.observation_space
        )
        return env
