import numpy as np
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        #self.fc2 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []  # contains the last capacity transitions

    def push(self, event):  # append a new event in the memory and make sure memory has 'capacity' events
        # event is a tuple of 4 elements: last state(st), new state(st+1), last
        # action(at), last reward(Rt)
        self.memory.append(event)  # append new event in memory
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning


class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):  # AI learns which action to select
        probs = F.softmax(
            self.model(
                Variable(
                    state,
                    volatile=True)) *
            75)  # temperature parameter = 75
        # higher is the temperature parameter, higher the probablity of the maximum q value to win
        # to deactivate the AI put temperature parameter = 0
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        # We need the next outputs because of the target which is
        # last reward + gamma * max of Q values wrt actions for the next state
        # max(1)[0] refers to max of Q values of next state (index = 0)
        # wrt actions (index = 1)
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)  # back propagation
        self.optimizer.step()  # this will update the weights

    def update(self, reward, new_signal):  # update all elements of transition and select transition
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # transition is composed of st, st+1, rt and at
        # we already have st, rt and at and we just got st+1 at line 65
        self.memory.push((self.last_state, new_state, torch.LongTensor(
            [int(self.last_action)]), torch.Tensor([self.last_reward])))
        # st = last_state, st+1 = new_state, at = last_action, rt = last_reward
        # at and rt are converted to torch tensors
        action = self.select_action(new_state)
        # if the number of transitions > 100, learn
        if len(self.memory.memory) > 100:
            # here 1st memory = object of ReplayMemory and 2nd memory =
            # memory[]
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
                100)
            self.learn(
                batch_state,
                batch_next_state,
                batch_reward,
                batch_action)
        # now update the last action, last state, last reward
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(), },
                   'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> Loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done")
        else:
            print("No checkpoint found")
