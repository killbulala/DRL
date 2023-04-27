# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/27 10:02
@Auth ： killbulala
@File ： 02_render2gif.py
@IDE  ： PyCharm
"""
import gym
import time
import matplotlib.pyplot as plt
from matplotlib import animation

env = gym.make('CartPole-v0')
env.reset()
terminated = False
score = 0
frames = []

while not terminated:
    frames.append(env.render(mode='rgb_array'))
    action = env.action_space.sample()
    obs, reward, terminated, info = env.step(action)
    score += reward
    time.sleep(0.1)


def frame2gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72, frames[0].shape[0]/72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save('gif_cartpole.gif')


env.close()
print(f'total score:{score}')

frame2gif(frames)

