# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/28 11:37
@Auth ： killbulala
@File ： 07_sarsa.py
@IDE  ： PyCharm
policy iteration
    policy gradient
value iteration
    sarsa
    q-learning
Rt : immediate reward
Gt : Rt+1 + γRt+2 + γ^2Rt+3 + '''   total reward
bellman equation 贝尔曼方程
    state value function  状态价值函数
    action value function 动作价值函数
MDP: markov decision process
马尔可夫性 : 下一时刻的状态与上一时刻的状态相关,与之前的状态无关(bellman equation成立的前提条件)
𝑄𝜋(𝑠,𝑎)) : 状态s下发生a动作的价值函数
    建立初始Q-table
    通过sarsa算法迭代更新Q-table

epsilon-greedy (explore, exploit)
    通过随机数与epsilon关系确定是探索还是利用
    学习初期尽量多选择explore模式, 随之学习进行要更关注exploit模式
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

"""maze"""


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


"""agent"""


class Agent:
    def __init__(self):
        self.action_space = list(range(4))
        self.theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                                 [np.nan, 1, np.nan, 1],  # s1
                                 [np.nan, np.nan, 1, 1],  # s2
                                 [1, np.nan, np.nan, np.nan],  # s3
                                 [np.nan, 1, 1, np.nan],  # s4
                                 [1, np.nan, np.nan, 1],  # s5
                                 [np.nan, 1, np.nan, np.nan],  # s6
                                 [1, 1, np.nan, 1]]  # s7
                                )
        self.pi = self._cvt_theta_to_pi()
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

    def choose_action(self, s):
        # eps, explore  eps的概率进行开发探索
        if np.random.rand() < self.eps:  # np.random.rand() 返回0-1均匀分布随机样本
            action = np.random.choice(self.action_space, p=self.pi[s, :])
        else:
            # 1-eps, exploit  1-eps的概率进行利用
            action = np.nanargmax(self.Q[s, :])
        return action

    def sarsa(self, s, a, r, s_next, a_next):
        """
        理想情况：
            𝑄(𝑠𝑡,𝑎𝑡)=𝑅𝑡+1+𝛾𝑄(𝑠𝑡+1,𝑎𝑡+1)
        偏差估计td(temporal difference error):
            td = 𝑅𝑡+1+𝛾𝑄(𝑠𝑡+1,𝑎𝑡+1) - 𝑄(𝑠𝑡,𝑎𝑡)
        final update equation:
            𝑄(𝑠𝑡,𝑎𝑡)=𝑄(𝑠𝑡,𝑎𝑡)+𝜂⋅[𝑅𝑡+1+𝛾𝑄(𝑠𝑡+1,𝑎𝑡+1)−𝑄(𝑠𝑡,𝑎𝑡)]
            𝑠𝑡,𝑎𝑡,𝑟𝑡+1,𝑠𝑡+1,𝑎𝑡+1   -> sarsa
        折扣（discount factor， 𝛾）: 折扣因子 有助于缩短步数,也可以更快的结束任务
        :param s:
        :param a:
        :param r:
        :param s_next:
        :param a_next:
        :return:
        """
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])


"""Sarsa (update  𝑄𝜋(𝑠,𝑎))"""
maze = MazeEnv()
agent = Agent()
epoch = 0
threshold = 1e-5
while True:
    # 保存上一个Q-table
    pre_Q = np.nanmax(agent.Q, axis=1)
    state = maze.reset()
    action = agent.choose_action(state)
    trajectory = [[state, np.nan]]
    while True:
        trajectory[-1][1] = action
        state_next, reward, done, info = maze.step(action)
        trajectory.append([state_next, np.nan])
        if done:
            action_next = np.nan
        else:
            action_next = agent.choose_action(state_next)
        agent.sarsa(state, action, reward, state_next, action_next)
        if done:
            break
        else:
            action = action_next
            state = state_next
    # trajectory, agent.Q-table 计算两次偏差
    td = np.sum(np.abs(np.nanmax(agent.Q, axis=1) - pre_Q))
    epoch += 1
    # 随着学习进行 将epsilon降低 使得更加关注利用模式
    agent.eps /= 2
    print(epoch, td, len(trajectory))
    # 根据学习轮数和偏差大小结束学习
    if epoch > 50 or td < threshold:
        break
