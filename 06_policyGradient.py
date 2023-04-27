# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/27 21:53
@Auth ： killbulala
@File ： 06_policyGradient.py
@IDE  ： PyCharm

θ -> π 的转换方式:
    naive简单: 均值计算概率
    general : softmax计算概率
policy gradient: 策略梯度
    算法思想:
        基于给定θ概率, 选择π策略
        完成一次trajectory
        根据trajectory利用转换方法计算新的θ概率, 选择新的π策略
        循环迭代 直到θ(end-1)和θ(end)之间误差小于threshold   误差越来越小
        θ0 -> π0 -> θ1 -> π1 -> ..... -> θ(end) -> env.reset() -> π(end)
"""
import numpy as np
import gym
import matplotlib.pyplot as plt

"""env"""
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
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    right=False,
    left=False,
    labelbottom=False,
    labelleft=False
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
        if self.state == 8:
            done = True
        # 对齐gym
        return self.state, 1, done, {}


# 动作策略选择，基于当前环境的状态
class Agent:
    def __init__(self):
        self.actions = list(range(4))
        self.theta_0 = np.array([[np.nan, 1, 1, np.nan],  # s0
                                 [np.nan, 1, np.nan, 1],  # s1
                                 [np.nan, np.nan, 1, 1],  # s2
                                 [1, np.nan, np.nan, np.nan],  # s3
                                 [np.nan, 1, 1, np.nan],  # s4
                                 [1, np.nan, np.nan, 1],  # s5
                                 [np.nan, 1, np.nan, np.nan],  # s6
                                 [1, 1, np.nan, 1]]  # s7
                                )
        self.theta = self.theta_0
        # self.pi = self._cvt_theta_0_to_pi(self.theta_0)
        self.pi = self._softmax_cvt_theta_to_pi()
        self.eta = 0.1

    def _cvt_theta_to_pi(self):
        m, n = self.theta.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, :] = self.theta[r, :] / np.nansum(self.theta[r, :])
        return np.nan_to_num(pi)

    # softmax计算概率
    def _softmax_cvt_theta_to_pi(self, beta=1.):
        m, n = self.theta.shape
        pi = np.zeros((m, n))
        exp_theta = np.exp(self.theta * beta)
        for r in range(m):
            pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])
        return np.nan_to_num(pi)

    # 基于trajectory
    def update_theta(self, trajectory):
        T = len(trajectory) - 1
        m, n = self.theta.shape
        delta_theta = self.theta.copy()
        for i in range(m):
            for j in range(n):
                if not (np.isnan(self.theta_0[i, j])):
                    sa_i = [sa for sa in trajectory if sa[0] == i]
                    sa_ij = [sa for sa in trajectory if (sa[0] == i and sa[1] == j)]
                    N_i = len(sa_i)
                    N_ij = len(sa_ij)
                    delta_theta[i, j] = (N_ij - self.pi[i, j] * N_i) / T
        self.theta = self.theta + self.eta * delta_theta

    def update_pi(self):
        self.pi = self._softmax_cvt_theta_to_pi()

    def choose_action(self, state):
        # 依据概率分布 π/θ 选择 action
        action = np.random.choice(self.actions, p=self.pi[state, :])  # np.random.choice(按照p概率的大小在待选列表抽样)
        return action


"""进行一次策略更新"""
# env = MazeEnv()
# agent = Agent()
# state = env.state
# trajectory = [[state, np.nan]]
# while True:
#     action = agent.choose_action(state)
#     trajectory[-1][1] = action
#     state, reward, done, _ = env.step(action)
#     trajectory.append([state, np.nan])
#     if state == 8 or done:
#         break
# print(f'初始的θ -> ')
# print(agent.theta)
# print(f'初始的π -> ')
# print(agent.pi)
# print(f'本次trajectory长度:{len(trajectory)}')
# # 根据trajectory进行θ更新
# agent.update_theta(trajectory)
# # 根据新的θ进行π策略更新
# agent._softmax_cvt_theta_to_pi()
# print(f'一次trajectory后的θ -> ')
# print(agent.theta)
# print(f'一次trajectory后的π -> ')
# print(agent.pi)


"""完整的policy gradient"""
# end_threshold = 1e-4   # threshold  10^-4
# agent = Agent()
# env = MazeEnv()
# while True:
#     # 不断地从初始状态出发，产生一次 trajectory
#     state = env.reset()
#     trajectory = [[state, np.nan]]
#     while True:
#         action = agent.choose_action(state)
#         trajectory[-1][1] = action
#         state, reward, done, _ = env.step(action)
#         trajectory.append([state, np.nan])
#         if state == 8 or done:
#             break
#     # 更新 theta
#     agent.update_theta(trajectory)
#     pi = agent.pi.copy()
#     # 更新 pi
#     agent.update_pi()
#     # L1范数定义两次概率偏差
#     delta = np.sum(np.abs(agent.pi - pi))
#     print(len(trajectory), delta)
#     if delta < end_threshold:
#         break
