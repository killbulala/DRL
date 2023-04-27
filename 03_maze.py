# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/27 10:29
@Auth ： killbulala
@File ： 03_maze.py
@IDE  ： PyCharm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

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
# plt.show()


# agent policy  π
theta = np.array([
    # [0, 1, 2, 3]  =>  [↑, →, ↓, ←]
    [np.nan, 1, 1, np.nan],       # S0
    [np.nan, 1, np.nan, 1],       # s1
    [np.nan, np.nan, 1, 1],       # s2
    [1, np.nan, np.nan, np.nan],  # s3
    [np.nan, 1, 1, np.nan],       # s4
    [1, np.nan, np.nan, 1],       # s5
    [np.nan, 1, np.nan, np.nan],  # s6
    [1, 1, np.nan, 1]             # s7
])


def theta2pi(cvt_theta):
    m, n = cvt_theta.shape
    pi = np.zeros(shape=(m, n))
    for row in range(m):
        pi[row, :] = cvt_theta[row, :] / np.nansum(cvt_theta[row, :])
    return np.nan_to_num(pi)


action_space = list(range(4))


def step(state, action):
    if action == 0:
        state -= 3
    if action == 1:
        state += 1
    if action == 2:
        state += 3
    if action == 3:
        state -= 1
    return state


state = 0
action_history = []
state_history = [state]
while True:
    action = np.random.choice(action_space, p=theta2pi(theta)[state, :])
    state = step(state, action)
    if state == 8:
        state_history.append(8)
        break
    action_history.append(action)
    state_history.append(state)

# print(action_history)
# print(state_history)
print(len(state_history))


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
plt.rcParams['animation.ffmpeg_path'] = r'D:\soft\ffmpeg\ffmpeg-5.1.2-full_build\bin\ffmpeg.exe'
writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'))
anim.save('maze.mp4', writer=writer)
# HTML(anim.to_jshtml())   # jupyter
