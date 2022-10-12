import typing
import numpy as np
from experience import ExperienceReplay, Experience
import torch
from torch import nn, optim
from torch.nn import functional as F


class Agent:

    def __init__(self, state_size,
                 action_size,
                 num_units,
                 optimizer,
                 batch_size,
                 buffer_size,
                 decay: typing.Callable[[int], float],
                 alpha,
                 gamma,
                 update_rate,
                 seed,
                 checkpoint=None):

        self.state_size_ = state_size
        self.action_size_ = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed is None:
            self.random_state_ = np.random.RandomState()
        else:
            self.random_state_ = np.random.RandomState(seed)

        if seed is not None:
            torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.memory = ExperienceReplay(batch_size, buffer_size, self.random_state_)
        self.decay_ = decay
        self.alpha_ = alpha
        self.gamma_ = gamma

        self.update_rate = update_rate
        if checkpoint is None:
            self.online_q = self.initialize_q(num_units)
            self.optimizer = optimizer(self.online_q.parameters())
        if checkpoint is not None:
            self.online_q.load_state_dict(checkpoint['online-q'])
            self.optimizer.load_state_dict(checkpoint['optim'])
        self.target_q = self.initialize_q(num_units)
        self.target_q.load_state_dict(self.online_q.state_dict())
        self.online_q.to(self.device)
        self.target_q.to(self.device)

        self.num_epochs = 0
        self.num_step = 0

    def initialize_q(self, num_units):
        net = nn.Sequential(nn.Linear(in_features=self.state_size_, out_features=num_units),
                            nn.ReLU(),
                            nn.Linear(in_features=num_units, out_features=num_units),
                            nn.ReLU(),
                            nn.Linear(in_features=num_units, out_features=self.action_size_))
        return net

    def action(self, state):
        tensor = (torch.from_numpy(state).unsqueeze(dim=0)).to(self.device)

        if not self.is_wise():
            action = self.random_state_.randint(self.action_size_)
        else:
            epsilon = self.decay_(self.num_epochs)
            if self.random_state_.random() < epsilon:
                action = self.random_state_.randint(self.action_size_)
            else:
                action = self.online_q(tensor).argmax().cpu().item()
        return action

    def update(self, states, rewards, dones, gamma, net1, net2):
        null, actions = net1(states).max(dim=1, keepdim=True)
        vals = self.actions_being_judged(states, actions, rewards, dones, gamma, net2)
        return vals

    def actions_being_judged(self, states, actions, rewards, dones, gamma, net):
        next_q = net(states).gather(dim=1, index=actions)
        vals = rewards + (gamma * next_q * (1 - dones))
        return vals

    def learn(self, experiences):

        states, actions, rewards, next_states, done = (torch.Tensor(np.array(i)).to(self.device) for i in zip(*experiences))

        actions = (actions.long().unsqueeze(dim=1))
        rewards = rewards.unsqueeze(dim=1)
        dones = done.unsqueeze(dim=1)

        target_vals = self.update(next_states, rewards, dones, self.gamma_, self.online_q, self.target_q)

        online_vals = self.online_q(states).gather(dim=1, index=actions)

        loss = F.mse_loss(online_vals, target_vals)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for p1, p2 in zip(self.target_q.parameters(), self.online_q.parameters()):
            p1.data.copy_(self.alpha_ * p2.data + (1 - self.alpha_) * p1.data)

    def is_wise(self):
        return ExperienceReplay.batch_length(self.memory) >= self.memory.batch_size_

    def step(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.memory.append(experience)

        if done:
            self.num_epochs += 1
        else:
            self.num_step += 1

            if self.num_step % self.update_rate == 0 and self.is_wise():
                experience = self.memory.sample()
                self.learn(experience)
