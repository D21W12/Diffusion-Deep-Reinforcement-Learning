import gymnasium as gym
from tqdm import tqdm

from .base_loop import Loop
from .evaluation_loop import EvaluationLoop
from ...agents import Agent
from project.util.evaluator import Evaluator


class ExperimentLoop(EvaluationLoop):

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            evaluator: Evaluator,
            sigma_noise: float
    ) -> None:
        super().__init__(env=env, agent=agent, evaluator=evaluator)
        self._sigma_noise = sigma_noise

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
                    sigma_noise=self._sigma_noise
                )

                cum_reward, t, n = 0, 0, n + 1
                s, info = self._env.reset()

                progress_bar.update(1)

            else:
                s = s_prime
