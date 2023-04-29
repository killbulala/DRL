# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/29 12:00
@Auth ： killbulala
@File ： 08_q-learning.py
@IDE  ： PyCharm

sarsa: 𝑄(𝑠𝑡,𝑎𝑡)=𝑄(𝑠𝑡,𝑎𝑡)+𝜂⋅(𝑅𝑡+1+𝛾𝑄(𝑠𝑡+1,𝑎𝑡+1)−𝑄(𝑠𝑡,𝑎𝑡))
    基于𝑠𝑡选择𝑎𝑡,产生𝑅𝑡+1和𝑠𝑡+1, 依赖于𝑠𝑡+1选择的𝑎𝑡+1进行迭代
    策略依赖型（on） : 依赖基于𝑠𝑡+1状态下选择的𝑎𝑡+1
    参数: 𝑠𝑡,𝑎𝑡, 𝑅𝑡+1, 𝑠𝑡+1,𝑎𝑡+1
q_learning: 𝑄(𝑠𝑡,𝑎𝑡)=𝑄(𝑠𝑡,𝑎𝑡)+𝜂⋅(𝑅𝑡+1+𝛾max𝑎𝑄(𝑠𝑡+1,𝑎)−𝑄(𝑠𝑡,𝑎𝑡))
    基于𝑠𝑡选择𝑎𝑡,产生𝑅𝑡+1和𝑠𝑡+1, 依赖于𝑠𝑡+1选择价值最大的a进行迭代, 不依赖𝑎𝑡+1
    策略关闭型（off）: 基于𝑠𝑡+1状态下q值最大的a,不依赖于𝑎𝑡+1
    参数: 𝑠𝑡,𝑎𝑡, 𝑅𝑡+1, 𝑠𝑡+1
"""
import numpy as np
import gym
import matplotlib.pyplot as plt

"""maze env"""
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


class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

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
        reward = 0
        if self.state == 8:
            done = True
            reward = 1
        # state, reward, done, _
        return self.state, reward, done, {}


class Agent:
    def __init__(self):
        self.action_space = list(range(4))
        self.theta_0 = np.asarray([[np.nan, 1, 1, np.nan],  # s0
                                   [np.nan, 1, np.nan, 1],  # s1
                                   [np.nan, np.nan, 1, 1],  # s2
                                   [1, np.nan, np.nan, np.nan],  # s3
                                   [np.nan, 1, 1, np.nan],  # s4
                                   [1, np.nan, np.nan, 1],  # s5
                                   [np.nan, 1, np.nan, np.nan],  # s6
                                   [1, 1, np.nan, 1]]  # s7
                                  )
        self.pi = self._cvt_theta_to_pi()
        # self.pi = self._softmax_cvt_theta_to_pi()
        # self.theta = self.theta_0

        self.Q = np.random.rand(*self.theta_0.shape) * self.theta_0
        self.eta = 0.1
        self.gamma = 0.9
        self.eps = 0.5

    def _cvt_theta_to_pi(self):
        m, n = self.theta_0.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, :] = self.theta_0[r, :] / np.nansum(self.theta_0[r, :])
        return np.nan_to_num(pi)

    # def _softmax_cvt_theta_to_pi(self, beta=1.):
    #     m, n = self.theta.shape
    #     pi = np.zeros((m, n))
    #     exp_theta = np.exp(self.theta*beta)
    #     for r in range(m):
    #         pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])
    #     return np.nan_to_num(pi)

    def get_action(self, s):
        # eps, explore
        if np.random.rand() < self.eps:
            action = np.random.choice(self.action_space, p=self.pi[s, :])
        else:
            # 1-eps, exploit
            action = np.nanargmax(self.Q[s, :])
        return action

    def sarsa(self, s, a, r, s_next, a_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])

    def q_learning(self, s, a, r, s_next):
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * np.nanmax(self.Q[s_next, :]) - self.Q[s, a])


maze = MazeEnv()
agent = Agent()
epoch = 0
while True:
    pre_Q = np.nanmax(agent.Q, axis=1)
    state = maze.reset()
    action = agent.get_action(state)
    trajectory = [[state, np.nan]]
    while True:
        # state, action
        trajectory[-1][1] = action
        state_next, reward, done, _ = maze.step(action)
        # state_next, action_next
        trajectory.append([state_next, np.nan])
        if done:
            action_next = np.nan
        else:
            action_next = agent.get_action(state_next)
        # q-learning update
        agent.q_learning(state, action, reward, state_next)
        if done:
            break
        else:
            action = action_next
            state = maze.state

    # trajectory, agent.Q
    td = np.sum(np.abs(np.nanmax(agent.Q, axis=1) - pre_Q))
    epoch += 1
    agent.eps /= 2
    print(epoch, td, len(trajectory))
    if epoch > 50 or td < 1e-5:
        break
