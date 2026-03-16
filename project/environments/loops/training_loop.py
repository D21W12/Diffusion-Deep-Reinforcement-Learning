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

    def run(self, frames: int, print_episode_length: bool = False):

        terminated = True

        elapsed_episodes = 0
        elapsed_episode_frames = 0

        for _ in tqdm(range(frames), desc="Frames: "):

            if terminated:
                s, info = self._env.reset()
                tqdm.write(f"Episode {elapsed_episodes} complete ({elapsed_episode_frames} frames)")
                elapsed_episodes += 1
                elapsed_episode_frames = 0

            a = self._agent.select_action(s=s)
            s_prime, reward, terminated, truncated, info = self._env.step(a)

            self._agent.observe(s=s, a=a, r=reward, s_prime=s_prime, t=terminated)
            self._agent.update()

            s = s_prime

            elapsed_episode_frames += 1
