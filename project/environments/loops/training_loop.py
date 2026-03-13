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

    def run(self, frames: int):

        e = True

        for _ in tqdm(range(frames), desc="Frames: "):

            if e:
                s, info = self._env.reset()

            a = self._agent.select_action(s=s)
            s_prime, r, e, _, _ = self._env.step(a)

            self._agent.observe(s=s, a=a, r=r, s_prime=s_prime)
            self._agent.update()

            s = s_prime
