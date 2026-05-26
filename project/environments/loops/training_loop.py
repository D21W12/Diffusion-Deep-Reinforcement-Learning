from typing import Callable

import gymnasium as gym

from tqdm import tqdm

from .base_loop import Loop
from ...agents import Agent
from ...util.weights_and_biases import WandB


class TrainingLoop(Loop):

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
            wandb: bool = True,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(env=env, agent=agent)

        self._wandb = WandB(project="DQN", *args, **kwargs) if wandb else None

    def run(
            self,
            frames: int,
            checkpoint: int | None = None,
            checkpoint_callback: Callable | None = None,
    ):

        s, info = self._env.reset()
        cum_reward = ts = 0

        for step in tqdm(range(frames), desc="Frames: "):

            a = self._agent.select_action(s=s)
            s_prime, reward, terminated, truncated, info = self._env.step(a)
            done = terminated or truncated

            self._agent.observe(s=s, a=a, r=reward, s_prime=s_prime, t=done)
            self._agent.update()

            s = s_prime

            ts += 1
            cum_reward += reward

            if (checkpoint is not None) and ((step + 1) % checkpoint == 0):
                checkpoint_callback()

            if done:
                s, info = self._env.reset()
                if self._wandb:
                    self._wandb.log(
                        cumulative_reward=cum_reward,
                        average_reward=cum_reward / ts,
                        episode_length=ts,
                    )
                cum_reward = ts = 0

        if self._wandb:
            self._wandb.finish()