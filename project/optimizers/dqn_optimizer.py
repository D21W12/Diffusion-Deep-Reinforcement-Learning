import gymnasium as gym
import torch

from tqdm import trange

from ..agents import DQNAgent


class DQNOptimizer:
    # TODO: implement frame skips
    # TODO: make loops based on frames instead of episodes

    def __init__(
            self,
            env: gym.Env,
            agent: DQNAgent,
            update_frequency: int = 4
    ) -> None:

        self._env = env
        self._agent = agent
        self._update_frequency = update_frequency

        self._step = 0

    def optimize(self, episodes: int):

        for _ in trange(episodes, desc="Episodes: "):

            s, info = self._env.reset()
            e = False

            while not e:

                a = self._agent.select_action(s=s)
                s_prime, r, e, _, _ = self._env.step(a)

                self._agent.observe(a=a, r=r, s_prime=s_prime)

                self._agent.update()

                s = s_prime
