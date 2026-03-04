import gymnasium as gym
import ale_py
from time import sleep

from project.environments import BaseEnvWrapper
import matplotlib.pyplot as plt

gym.register_envs(ale_py)

FPS = 60
VISUAL = True

env = gym.make('ALE/Breakout-v5', render_mode='human' if VISUAL else None)
env = BaseEnvWrapper(env, frame_stack_size=1)
obs, info = env.reset()

for i in range(4):
    sleep(1 / FPS)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

plt.imshow(obs.squeeze(0), cmap='grey')
plt.show()
env.close()
