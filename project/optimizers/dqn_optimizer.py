import gymnasium as gym

from tqdm import trange

from ..agents import DQNAgent


class DQNOptimizer:

    def __init__(
            self,
            env: gym.Env,
            agent: DQNAgent
    ) -> None:

        self._env = env
        self._agent = agent

    def optimize(self, episodes: int):

        for _ in trange(episodes, desc="Episodes: "):

            s, info = self._env.reset()
            e = False

            while not e:

                a = self._agent.select_action(s=s)
                s_prime, r, e, _, _ = self._env.step(a)

                self._agent.observe(
                    a=a,
                    r=r,
                    s_prime=s_prime
                )

                s = s_prime
