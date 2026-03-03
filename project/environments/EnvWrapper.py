import gymnasium as gym


class EnvWrapper(gym.Wrapper):

    def __init__(
            self,
            env: gym.Env
    ) -> None:
        super().__init__(env)
