import gymnasium as gym

from project.agents import DQNAgent


class DQNOptimizer:

    def __init__(
            self,
            env: gym.Env,
            agent: DQNAgent
    ) -> None:

        self._env = env
        self._agent = agent

    def optimize(self, episodes: int):

        for _ in range(episodes):

            obs, info = self._env.reset()
            e = False

            while not e:

                pass



