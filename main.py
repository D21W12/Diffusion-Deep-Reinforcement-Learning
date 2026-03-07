import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt

from time import sleep

from project.environments import BaseWrapper


gym.register_envs(ale_py)

FPS = 60
VISUAL = True

env = gym.make('ALE/Breakout-v5', render_mode='human' if VISUAL else None)
env = BaseWrapper(env, frame_stack_size=4)
obs, info = env.reset()

print(env.observation_space.shape)

for i in range(180):
    sleep(1 / FPS)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs.shape)

plt.imshow(obs.squeeze(0), cmap='grey')
plt.show()
env.close()
