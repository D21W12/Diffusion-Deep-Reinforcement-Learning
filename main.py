import ale_py
import gymnasium as gym

from project.agents import DQNAgent
from project.environments import BaseWrapper
from project.environments.loops import TestingLoop


if __name__ == "__main__":

    gym.register_envs(ale_py)

    env = gym.make('ALE/Breakout-v5', render_mode='human' )
    env = BaseWrapper(env, frame_stack_size=4)

    agent = DQNAgent(
        train=False,
        n_actions=4,
        obs_shape=env.observation_space.shape
    )

    loop = TestingLoop(
        env=env,
        agent=agent
    )
    loop.run()
