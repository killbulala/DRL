# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/30 10:11
@Auth ： killbulala
@File ： 09_continuous2discrete.py
@IDE  ： PyCharm
场景中的状态或者观测值是连续的 不方便优化  需要将连续空间离散化
discrete:
    state -> action -> state
continuous:
    action -> box(4,) -> digitize -> id
"""
import gym
import numpy as np

env = gym.make('CartPole-v0')
env.reset()
action = env.action_space.sample()
# observation [ 0.03030827  0.24222113  0.00164348 -0.33564276]
observation, reward, done, info = env.step(action)
NUM_DIGITIZED = 6


# 分桶， 5个值，对应 6 个分段，即 6 个桶 (0, 1, 2, 3, 4, 5)
def bins(clip_min, clip_max, num_bins=NUM_DIGITIZED):
    return np.linspace(clip_min, clip_max, num_bins + 1)[1:-1]


# 按 6 进制映射将 4位 6 进制数映射为 id，
def digitize_state(obs):
    pos, cart_v, angle, pole_v = obs
    digitized = [np.digitize(pos, bins=bins(-2.4, 2.4, NUM_DIGITIZED)),
                 np.digitize(cart_v, bins=bins(-3., 3, NUM_DIGITIZED)),
                 np.digitize(angle, bins=bins(-0.418, 0.418, NUM_DIGITIZED)),
                 np.digitize(pole_v, bins=bins(-2, 2, NUM_DIGITIZED))]
    # 3,1,2,4 (4位10进制数) = 4*10^0 + 2*10^1 + 1*10^2 + 3*10^3，最终的取值范围是 0-9999，总计 10^4 == 10000
    # a,b,c,d (4位6进制数) = d*6^0 + c*6^1 + b*6^2 + a*6^3，最终的取值范围是 0-`5555`(1295)，总计 6^4 == 1296
    w = len(obs) - 1
    ind = sum([d * (NUM_DIGITIZED ** (w - i)) for i, d in enumerate(digitized)])
    return ind


print(digitize_state(observation))
