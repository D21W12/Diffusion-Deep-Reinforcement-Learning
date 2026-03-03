import gymnasium as gym


class BaseEnvWrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env
    ) -> None:
        super().__init__(env)
