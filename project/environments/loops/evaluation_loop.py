import gymnasium as gym
from tqdm import tqdm

from .base_loop import Loop
from ...agents import Agent
from project.util.evaluator import Evaluator


class EvaluationLoop(Loop):

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            evaluator: Evaluator
    ) -> None:
        super().__init__(env=env, agent=agent)

        self._evaluator = evaluator

    def run(self, episodes: int) -> None:
        s, info = self._env.reset()

        cum_reward = 0
        t = 0
        n = 0

        progress_bar = tqdm(desc="Episodes", total=episodes)

        while n < episodes:
            a = self._agent.select_action(s=s)
            s_prime, r, terminated, truncated, _ = self._env.step(a)
            done = terminated or truncated

            cum_reward += r
            t += 1

            if done:

                self._evaluator.record(
                    length=t,
                    reward=cum_reward,
                )

                cum_reward, t, n = 0, 0, n + 1
                s, info = self._env.reset()

                progress_bar.update(1)

            else:
                s = s_prime
