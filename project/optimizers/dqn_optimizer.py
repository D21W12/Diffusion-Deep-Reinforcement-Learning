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
            k: int = 4,
    ) -> None:

        self._env = env
        self._agent = agent

        self._k = k

        self._step = 0

    def optimize(self, frames: int):

        e = True
        a = None
        frame = 0

        for _ in trange(frames, desc="Frames: "):

            if e:
                s, info = self._env.reset()

            if frame % self._k == 0:
                a = self._agent.select_action(s=s)
                s_prime, r, e, _, _ = self._env.step(a)

                self._agent.observe(a=a, r=r, s_prime=s_prime)
                self._agent.update()

                s = s_prime

            else:
                s, _, e, _, _ = self._env.step(a)

            frames += 1
