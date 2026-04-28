import gymnasium as gym

from .base_loop import Loop
from ...agents import Agent
from project.evaluation import Evaluator


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

        while n < episodes:
            a = self._agent.select_action(s=s)
            s_prime, r, terminated, truncated, _ = self._env.step(a)
            done = terminated or truncated

            cum_reward += r
            t += 1

            if done:

                self._evaluator.record(
                    episode_length=t,
                    cum_reward=cum_reward,
                    avg_reward=cum_reward / t
                )

                cum_reward, t, n = 0, 0, n + 1
                s, info = self._env.reset()

            else:
                s = s_prime
