# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/27 9:25
@Auth ： killbulala
@File ： 01_初识强化环境.py
@IDE  ： PyCharm
Gym是什么?
    强化学习的环境库,根据不同的场景,选择不同的环境,大致分为离散和连续两种环境
"""
# gym==0.21.0
import gym

# 选择CartPole环境 倒立摆
env = gym.make('CartPole-v0')
# 将环境设置为初始状态
env.reset()
# 结束标志
terminated = False
# 得分
score = 0
# 简单显示倒立摆
while not terminated:
    env.render(mode='human')
    action = env.action_space.sample()
    obs, reward, terminated, info = env.step(action)
    print(obs)
    score += reward
env.close()
print(f'total score : {score}')

