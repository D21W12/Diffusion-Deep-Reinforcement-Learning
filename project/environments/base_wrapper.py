import gymnasium as gym


class BaseEnvWrapper(gym.Wrapper):
    """
    Base
    """

    def __init__(
            self,
            env: gym.Env,
            frame_stack_size: int = 4,
            use_torch: bool = True,
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
            use_torch=use_torch,
            grey_scale=grey_scale,
        )
        super().__init__(env)

    @staticmethod
    def _wrap_env(
            env: gym.Env,
            frame_stack_size: int,
            use_torch: bool,
            grey_scale: bool,
    ) -> gym.Env:

        if grey_scale: env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, frame_stack_size)
        if use_torch: env = gym.wrappers.NumpyToTorch(env)
        return env
