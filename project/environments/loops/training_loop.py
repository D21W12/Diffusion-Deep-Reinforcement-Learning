from typing import Callable

import gymnasium as gym

from tqdm import tqdm

from .base_loop import Loop
from ...agents import Agent


class TrainingLoop(Loop):

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ) -> None:
        super().__init__(env=env, agent=agent)

    def run(
            self,
            frames: int,
            checkpoint: int | None = None,
            checkpoint_callback: Callable | None = None,
    ):

        print(checkpoint)

        s, info = self._env.reset()

        for step in tqdm(range(frames), desc="Frames: "):

            a = self._agent.select_action(s=s)
            s_prime, reward, terminated, truncated, info = self._env.step(a)
            done = terminated or truncated

            self._agent.observe(s=s, a=a, r=reward, s_prime=s_prime, t=done)
            self._agent.update()

            s = s_prime

            if (checkpoint is not None) and ((step + 1) % checkpoint == 0):
                checkpoint_callback()

            if done:
                s, info = self._env.reset()
