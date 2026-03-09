import ale_py
import gymnasium as gym
import torch.mps

from project.environments import BaseWrapper
from project.agents import DQNAgent
from project.optimizers import DQNOptimizer

if __name__ == "__main__":

    PATH = "../weights/milkyway_test.pth"

    gym.register_envs(ale_py)

    env = gym.make('ALE/Breakout-v5')
    env = BaseWrapper(env, frame_stack_size=4)

    obs, info = env.reset()

    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Using: {device}")

    agent = DQNAgent(
        train=True,
        lr=1e-3,
        discount=0.99,
        replay_size=10000,
        n_actions=4,
        obs_shape=env.observation_space.shape
    ).to(device)
    agent.load(PATH)

    optimizer = DQNOptimizer(
        env=env,
        agent=agent,
    )

    optimizer.optimize(25_000)
    agent.save(PATH)
