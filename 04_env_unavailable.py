# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/27 19:12
@Auth ： killbulala
@File ： 04_env_unavailable.py
@IDE  ： PyCharm
构建maze环境类
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class MazeEnv:
    def __init__(self):
        self.fig = None
        self.line = None
        self.plt = None
        self.init_figure()
        self.theta = None
        self.init_theta()
        self.action_space = list(range(4))
        self.state = 0  # 初始状态
        self.action_history = []
        self.state_history = [self.state]

    def init_figure(self):
        """
        初始化 画布
        :return:
        """
        self.fig = plt.figure(figsize=(5, 5))
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
        plt.axis('off')
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False, top=False,
            right=False, left=False,
            labelbottom=False,
            labelleft=False
        )
        self.line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)
        self.plt = plt

    def init_theta(self):
        """
        agent policy  π
        构建maze环境中每个格子可移动位置
        :return:
        """
        self.theta = np.array([
            # [0, 1, 2, 3]  =>  [↑, →, ↓, ←]
            [np.nan, 1, 1, np.nan],  # S0
            [np.nan, 1, np.nan, 1],  # s1
            [np.nan, np.nan, 1, 1],  # s2
            [1, np.nan, np.nan, np.nan],  # s3
            [np.nan, 1, 1, np.nan],  # s4
            [1, np.nan, np.nan, 1],  # s5
            [np.nan, 1, np.nan, np.nan],  # s6
            [1, 1, np.nan, 1]  # s7
        ])

    def theta2pi(self, cvt_theta):
        """
        将可移动位置的矩阵转换成可选择策略的信息  进行概率选择
        :param cvt_theta:
        :return:
        """
        m, n = cvt_theta.shape
        pi = np.zeros(shape=(m, n))
        for row in range(m):
            pi[row, :] = cvt_theta[row, :] / np.nansum(cvt_theta[row, :])
        return np.nan_to_num(pi)

    def step(self, state, action):
        """
        动作空间  上下左右四个方向  移动后转态的位置变化信息
        :param state:
        :param action:
        :return:
        """
        if action == 0:
            state -= 3
        if action == 1:
            state += 1
        if action == 2:
            state += 3
        if action == 3:
            state -= 1
        return state

    def random_pi(self):
        """
        随机选择策略
        :return:
        """
        while True:
            action = np.random.choice(self.action_space, p=self.theta2pi(self.theta)[self.state, :])
            self.state = self.step(self.state, action)
            if self.state == 8:
                self.state_history.append(8)
                break
            self.action_history.append(action)
            self.state_history.append(self.state)

    def visual(self, name):
        """
        将选择路径信息可视化  支持gif和mp4
        :param name:
        :return:
        """

        def init():
            self.line.set_data([], [])
            return (self.line,)

        def animate(i):
            state = self.state_history[i]
            x = (state % 3) + 0.5
            y = 2.5 - int(state / 3)
            self.line.set_data(x, y)

        anim = animation.FuncAnimation(
            self.fig,
            animate,
            init_func=init,
            frames=len(self.state_history),
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
    M = MazeEnv()
    M.random_pi()
    M.visual('maze.mp4')

