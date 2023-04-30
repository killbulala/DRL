# -*- coding: utf-8 -*-
"""
@Time ： 2023/4/30 10:46
@Auth ： killbulala
@File ： 10_q-learningCartPole.py
@IDE  ： PyCharm
CartPole存活195轮是高端玩家水平
"""
import gym
import numpy as np


class Agent:
    def __init__(self, action_space, n_obs, eta=0.5, gamma=0.99, num=6):
        self.p = 0.5
        self.eta = eta
        self.gamma = gamma
        self.action_space = action_space
        self.num = num
        self.q_table = np.random.uniform(0, 1, size=(num ** n_obs, self.action_space.n))

    @staticmethod
    def _bins(clip_min, clip_max, num_bins):
        return np.linspace(clip_min, clip_max, num=num_bins)[1:-1]

    @staticmethod
    def _digitize_space(observation, num):
        position, cart_v, angle, angle_v = observation
        digitized = [
            np.digitize(position, bins=Agent._bins(-2.4, 2.4, num)),
            np.digitize(cart_v, bins=Agent._bins(-3, 3, num)),  # 经验值
            np.digitize(angle, bins=Agent._bins(-0.418, 0.418, num)),
            np.digitize(angle_v, bins=Agent._bins(-2, 2, num)),  # 经验值
        ]
        w = len(digitized) - 1
        ind = sum(d * num ** (w - i) for i, d in enumerate(digitized))
        # ind = sum([d * (num ** i) for i, d in enumerate(digitized)])
        return ind

    def q_learning(self, obs, action, reward, obs_next):
        obs_ind = Agent._digitize_space(obs, self.num)
        obs_next_ind = Agent._digitize_space(obs_next, self.num)
        self.q_table[obs_ind, action] = self.q_table[obs_ind, action] + \
                                        self.eta * (reward + max(self.q_table[obs_next_ind, :]) - self.q_table[obs_ind, action])

    def choose_action(self, obs, episode):
        eps = self.p * 1 / (episode + 1)
        obs_ind = self._digitize_space(obs, self.num)
        if np.random.random() < eps:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.q_table[obs_ind, :])
        return action


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.reset()
    action_space = env.action_space
    n_obs = env.observation_space.shape[0]
    agent = Agent(action_space, n_obs)
    max_episodes = 1000
    max_steps = 500
    level = 195
    continue_success_episodes = 0
    q_learning_finish_flag = False

    for episode in range(max_episodes):
        obs = env.reset()
        for step in range(max_steps):
            action = agent.choose_action(obs, episode)
            obs_next, _, done, info = env.step(action)
            if done:
                if step < level:
                    reward = -1
                    continue_success_episodes = 0
                else:
                    reward = 1
                    continue_success_episodes += 1
            else:
                reward = 0
            agent.q_learning(obs, action, reward, obs_next)
            if done:
                print(f'episode : {episode}, finish step:{step}')
                break
            obs = obs_next

        if q_learning_finish_flag:
            break
        if continue_success_episodes >= 10:
            q_learning_finish_flag = True
            print(f'continue success(step>195) more than 10 times')
