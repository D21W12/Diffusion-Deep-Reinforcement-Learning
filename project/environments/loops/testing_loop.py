import gymnasium as gym

from .base_loop import Loop
from ...agents import Agent


class TestingLoop(Loop):

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ) -> None:
        super().__init__(env=env, agent=agent)

    def run(self):
        s, info = self._env.reset()
        while True:
            a = self._agent.select_action(s=s)
            s_prime, r, terminated, truncated, _ = self._env.step(a)
            done = terminated or truncated

            if done:
                s, info = self._env.reset()
            else:
                s = s_prime
