# -*- coding: utf-8 -*-
"""
@Time ： 2023/6/1 15:05
@Auth ： killbulala
@File ： dqn.py
@IDE  ： PyCharm
"""
import gym
import torch
import random
from torch import nn
from collections import namedtuple
import matplotlib.pyplot as plt


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'state_next'))


# memory replay
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def push(self, state, action, reward, state_next):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, reward, state_next)
        self.index = (self.index + 1) % self.capacity


# Deep Q-learning network
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.rl1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.rl2 = nn.ReLU()
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        fc1_out = self.rl1(self.fc1(x))
        fc2_out = self.rl2(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)
        return fc3_out


# Agent
class Agent:
    def __init__(self, n_states, n_actions, capacity=20000, batch_size=100):
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = ReplayMemory(capacity=capacity)
        self.model = DQN(self.n_states, self.n_actions)
        self.loss = nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        self.gamma = 0.9

    def choose_action(self, state, episode, train_flag=True):
        """
        单步进行  state [1, 4]
        """
        if train_flag:
            eps = 0.5 * 1. / (episode + 1)
            if random.random() < eps:  # random.random() ：随机生成一个[0,1)内的浮点数。
                action = torch.tensor([random.randrange(self.n_actions)], dtype=torch.int).unsqueeze(dim=0)
            else:
                self.model.eval()
                with torch.no_grad():
                    action = self.model(state).max(dim=1)[1].view(1, 1)
            return action
        else:
            with torch.no_grad():
                action = self.model(state).max(dim=1)[1].view(1, 1)
            return action

    def memorize(self, state, action, reward, state_next):
        self.memory.push(state, action, reward, state_next)

    def q_function(self):
        if len(self.memory) < self.batch_size:
            return
        # build batch
        batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*batch))
        batch_state = torch.cat(batch.state)
        batch_action = torch.cat(batch.action)
        batch_reward = torch.cat(batch.reward)
        batch_state_next_without_None = torch.cat([s for s in batch.state_next if s is not None])

        self.model.eval()

        # pred_Q 功能： 在dim维度上，按照indexs所给的坐标选择元素，返回一个和indexs维度相同大小的tensor。
        pred_Q = self.model(batch_state).gather(dim=1, index=batch_action)
        mask_without_None = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.state_next)))
        Q_s1a = torch.zeros(self.batch_size)
        Q_s1a[mask_without_None] = self.model(batch_state_next_without_None).max(dim=1)[0].detach()
        # true_Q
        true_Q = (batch_reward + self.gamma * Q_s1a).unsqueeze(dim=1)

        # y_true = r1 + self.gamma * torch.max(self.eval_net(s1).detach(), dim=1)[0].view(self.batch_size, -1)
        # y_pred = self.eval_net(s0).gather(1, a0)

        self.model.train()
        loss = self.loss(pred_Q, true_Q)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


if __name__ == '__main__':
    flag = 'predict'
    assert flag in ['train', 'predict']
    if flag == 'train':
        """train"""
        env = gym.make('CartPole-v0')
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        agent = Agent(n_states, n_actions)
        high_level = 195
        complete_episodes = 0
        finished_flag = False
        max_episodes = 100
        max_steps = 250

        visual_steps = []
        for episode in range(max_episodes):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)

            for step in range(max_steps):
                action = agent.choose_action(state, episode)
                state_next, _, terminal, info = env.step(action.item())
                if terminal:
                    state_next = None
                    if step < high_level:
                        reward = torch.FloatTensor([-1.])
                        complete_episodes = 0
                    else:
                        reward = torch.FloatTensor([1.])
                        complete_episodes += 1
                else:
                    reward = torch.FloatTensor([0.])
                    state_next = torch.tensor(state_next, dtype=torch.float).unsqueeze(dim=0)

                # push memory
                agent.memorize(state, action, reward, state_next)
                # q-learning network
                agent.q_function()

                state = state_next

                if terminal:
                    print(f'episode : {str(episode+1)} - step : {str(step+1)}')
                    visual_steps.append(step)
                    break

                if complete_episodes >= 10:
                    finished_flag = True

                if finished_flag:
                    break
        plt.plot(visual_steps)
        plt.show()
        torch.save(agent.model.state_dict(), 'dqn_params.pth')

    """predict"""
    env = gym.make('CartPole-v0')
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = Agent(n_states, n_actions)
    # load params
    state_dict = torch.load('dqn_params.pth')
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
    terminal = False
    step = 0
    while not terminal:
        action = agent.choose_action(state, episode=0, train_flag=False)
        state_next, reward, terminal, _ = env.step(action.item())
        if terminal:
            break
        else:
            step += 1
            state = torch.tensor(state_next, dtype=torch.float32).unsqueeze(dim=0)
    print(f'death-step : {step}')




