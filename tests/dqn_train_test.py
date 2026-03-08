import ale_py
import gymnasium as gym

from project.environments import BaseWrapper
from project.agents import DQNAgent
from project.optimizers import DQNOptimizer

if __name__ == "__main__":

    gym.register_envs(ale_py)

    env = gym.make('ALE/Breakout-v5')
    env = BaseWrapper(env, frame_stack_size=4)

    obs, info = env.reset()

    agent = DQNAgent(
        lr=1e-3,
        discount=0.99,
        replay_size=10000,
        n_actions=4,
        obs_shape=env.observation_space.shape
    )
    agent.train()

    optimizer = DQNOptimizer(
        env=env,
        agent=agent,
    )

    optimizer.optimize(1000)