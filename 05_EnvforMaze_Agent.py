# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/27 20:01
@Auth ： killbulala
@File ： 05_EnvforMaze_Agent.py
@IDE  ： PyCharm
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0

    def step(self, action):
        if action == 0:
            self.state -= 3
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 3
        elif action == 3:
            self.state -= 1
        done = False
        if self.state == 8:
            done = True
        return self.state, 1, done, {}  # 类比gym返回值  state reward terminate information


class Agent:
    def __init__(self):
        self.actions = list(range(4))
        self.theta = np.asarray([[np.nan, 1, 1, np.nan],  # s0
                                 [np.nan, 1, np.nan, 1],  # s1
                                 [np.nan, np.nan, 1, 1],  # s2
                                 [1, np.nan, np.nan, np.nan],  # s3
                                 [np.nan, 1, 1, np.nan],  # s4
                                 [1, np.nan, np.nan, 1],  # s5
                                 [np.nan, 1, np.nan, np.nan],  # s6
                                 [1, 1, np.nan, 1]]  # s7
                                )

        self.pi = self.theta2pi(self.theta)

    def theta2pi(self, cvt_theta):
        m, n = cvt_theta.shape
        pi = np.zeros(shape=(m, n))
        for row in range(m):
            pi[row, :] = cvt_theta[row, :] / np.nansum(cvt_theta[row, :])
        return np.nan_to_num(pi)

    def random_choose_action(self, state):
        action = np.random.choice(self.actions, p=self.pi[state, :])
        return action


# maze画布
fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.plot([2, 3], [1, 1], color='red', linewidth=2)
plt.plot([0, 1], [1, 1], color='red', linewidth=2)
plt.plot([1, 1], [1, 2], color='red', linewidth=2)
plt.plot([1, 2], [2, 2], color='red', linewidth=2)

plt.text(0.5, 2.5, 'S0', size=14, ha='center')
plt.text(1.5, 2.5, 'S1', size=14, ha='center')
plt.text(2.5, 2.5, 'S2', size=14, ha='center')
plt.text(0.5, 1.5, 'S3', size=14, ha='center')
plt.text(1.5, 1.5, 'S4', size=14, ha='center')
plt.text(2.5, 1.5, 'S5', size=14, ha='center')
plt.text(0.5, 0.5, 'S6', size=14, ha='center')
plt.text(1.5, 0.5, 'S7', size=14, ha='center')
plt.text(2.5, 0.5, 'S8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')
# plt.axis('off')
plt.tick_params(axis='both', which='both',
                bottom=False, top=False,
                right=False, left=False,
                labelbottom=False, labelleft=False
                )
line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)


# 可视化函数
def visual(fig, line, state_history, name):
    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        state = state_history[i]
        x = (state % 3) + 0.5
        y = 2.5 - int(state / 3)
        line.set_data(x, y)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(state_history),
        interval=200,
        repeat=False
    )
    # 添加 ImageMagick ffmpeg 安装地址
    plt.rcParams['animation.convert_path'] = r'D:\soft\ImageMagick\ImageMagick-7.1.1-Q16\magick.exe'
    plt.rcParams['animation.ffmpeg_path'] = r'D:\soft\ffmpeg\ffmpeg-5.1.2-full_build\bin\ffmpeg.exe'
    if name.endswith('.gif'):
        writer = animation.ImageMagickWriter()
    else:
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'))
    anim.save(name, writer=writer)


if __name__ == '__main__':
    env = MazeEnv()
    env.reset()
    agent = Agent()
    done = False
    action_history = []
    state_history = [env.state]
    while not done:
        action = agent.random_choose_action(env.state)
        state, reward, done, info = env.step(action)
        action_history.append(action)
        state_history.append(state)
    visual(fig, line, state_history, name='maze.mp4')
