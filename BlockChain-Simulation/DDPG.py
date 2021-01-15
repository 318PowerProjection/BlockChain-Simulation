import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from params import tau, gamma, capacity, batch_size, update_iteration, actor_alpha, critic_alpha

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.maxSize = capacity
        self.ptr = 0

    def put(self, data):
        if len(self.buffer) == self.maxSize:
            self.buffer[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.maxSize
        else:
            self.buffer.append(data)

    def take(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        S, A, R, SS, D = [], [], [], [], []
        for it in ind:
            s, a, r, ss, d = self.buffer[it]
            S.append(s)
            A.append(a)
            R.append(r)
            SS.append(ss)
            D.append(d)
        return np.array(S), np.array(A), np.array(R), np.array(SS), np.array(D)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.sigmoid(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat((x, u), 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, seed):
        torch.manual_seed(seed)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_alpha)
        # actor的优化器

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_alpha)
        self.replay_buffer = ReplayBuffer()

        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)      # actor网络选择动作
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self, seed):
        for it in range(update_iteration):
            # Sample replay buffer
            s, a, r, ss, d = self.replay_buffer.take(batch_size)
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(ss).to(device)
            reward = torch.FloatTensor(r).reshape(-1, 1).to(device)
            done = torch.FloatTensor(1-d).reshape(-1, 1).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))    # 求TD target
            target_Q = reward + (done * gamma * target_Q).detach()                 # 求TD error

            # Get current Q estimate
            current_Q = self.critic(state, action)          # 求Q值

            # Compute critic loss
            critic_output = open('./Output/Loss/seed=%d/critic_loss.txt' % seed, 'a')
            critic_loss = F.mse_loss(current_Q, target_Q)
            print(self.num_training, critic_loss.cpu().detach().numpy().tolist(), file=critic_output)
            critic_output.close()

            # Optimize the critic
            self.critic_optimizer.zero_grad()       # 梯度下降基础操作
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_output = open('./Output/Loss/seed=%d/actor_loss.txt' % seed, 'a')
            actor_loss = -self.critic(state, self.actor(state)).mean()
            print(self.num_training, actor_loss.cpu().detach().numpy().tolist(), file=actor_output)
            actor_output.close()

            # Optimize the actor
            self.actor_optimizer.zero_grad()        # 梯度初始化为0
            actor_loss.backward()                   # 反向传播求梯度
            self.actor_optimizer.step()             # 更新所有参数

            # Update the frozen target models
            # soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.num_training += 1

    def save(self, seed, episode):
        torch.save(self.actor.state_dict(), './Output/Model/seed=%d/actor_%d.pth' % (seed, episode))

    def load(self, seed, episode):
        self.actor.load_state_dict(torch.load('./Output/Model/seed=%d/actor_%d.pth' % (seed, episode)))
