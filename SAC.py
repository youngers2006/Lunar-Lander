import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import itertools

class StateNorm:
    def __init__(self, state_size):
        self.mean = torch.zeros(size=state_size)
        self.variance = torch.zeros(size=state_size)
        self.count = 1e-4
        self.ready = True

    def update(self, state_batch):
        if self.ready:
            batch_mean = torch.mean(state_batch, dim=0)
            batch_var = torch.var(state_batch, dim=0)
            batch_count = state_batch.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        self.count += count
        new_mean = self.mean + delta * count / self.count
        m_a = self.var * self.count
        m_b = var * count
        new_var = (m_a + m_b + delta ** 2 * self.count * count / self.count) / self.count
        self.mean = new_mean
        self.var = new_var

    def normalise(self, states):
        states_norm = (states - self.mean) / (torch.sqrt(self.var) + 1e-8)
        return states_norm

class Dataset:
    def __init__(self, buffer_size, batch_size, train_epochs, Q1_optimiser, Q2_optimiser, actor_optimiser, tau, gamma, seed, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.epochs = train_epochs
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.sample_count = 0
        self.opt_Q1 = Q1_optimiser
        self.opt_Q2 = Q2_optimiser
        self.opt_actor = actor_optimiser
        self.device = device

    def rollout(self, num_rollouts, env, actor):
        state, _ = env.reset(seed=self.seed)
        for i in tqdm(range(num_rollouts), desc=f'Running Rollouts', leave=False):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action, _ = actor.act(state_tensor)
            state_, reward, terminated, truncated, _ = env.step(action)
            done = (terminated or truncated)
            self.buffer.append((state, action, reward, state_, done))
            state = state_
            self.sample_count += 1
            if done:
                self.seed += 1
                state, _ = env.reset(seed=self.seed)
                done = False

    def batch_buffer(self):
        total_length = len(self.buffer)
        for i in range(0, total_length, self.batch_size):
            stop_index = min(i + self.batch_size, total_length)
            batch = list(itertools.islice(self.buffer, i, stop_index))

            rb = torch.tensor(batch[2], dtype=torch.float32, device=self.device)
            sb = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=self.device)
            nsb = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=self.device)
            ab = torch.tensor(np.array(batch[1]), dtype=torch.float32, device=self.device)
            db = torch.tensor(np.array(batch[4]), dtype=torch.float32, device=self.device)
            yield sb, ab, rb, nsb, db

    def soft_update(self, target, main):
        with torch.no_grad():
            for target_param, main_param in zip(target.parameters(), main.parameters()):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
        return target
    
    def train_step(self, Q1, Q2, QT1, QT2, actor, state, action, reward, state_, done):
        Q_pred1 = Q1.forward(state, action)
        Q_pred2 = Q2.forward(state, action)
        action_, log_prob_ = actor(state_)
        Q_pred_targ1 = QT1.forward(state_, action_)
        Q_pred_targ2 = QT2.forward(state_, action_)
        Q_tensor_targ = torch.tensor([Q_pred_targ1, Q_pred_targ2], dtype=torch.float32, device=self.device)
        min_Q_targ = torch.min(Q_tensor_targ, dim=-1)
        y = reward + self.gamma * (1 - done) * (min_Q_targ - self.alpha * log_prob_)
        loss_Q_1 = (1 / self.batch_size) * torch.sum(torch.square(Q_pred1 - y))
        loss_Q_2 = (1 / self.batch_size) * torch.sum(torch.square(Q_pred2 - y))
        action_pred, log_prob = actor(state)
        Q_pred_1_1 = Q1.forward(state, action_pred)
        Q_pred_2_1 = Q1.forward(state, action_pred)
        Q_tensor = torch.tensor([Q_pred_1_1, Q_pred_2_1], dtype=torch.float32, device=self.device)
        min_Q = torch.min(Q_tensor, dim=-1)
        loss_actor = (1 / self.batch_size) * torch.sum(min_Q - self.alpha * log_prob)

        self.opt_Q1.zero_grad()
        loss_actor.backwards()
        self.opt_Q1.step()

        self.opt_Q2.zero_grad()
        loss_actor.backwards()
        self.opt_Q2.step()

        self.opt_actor.zero_grad()
        loss_actor.backwards()
        self.opt_actor.step()

        QT1 = self.soft_update(QT1, Q1)
        QT2 = self.soft_update(QT2, Q2)
        return Q1, Q2, QT1, QT2, actor

    def update_networks(self, Q1, Q2, QT1, QT2, actor):
        for epoch in range(self.epochs):
            for state, action, reward, state_, done in tqdm(self.batch_buffer(), desc=f'Training Networks, in epoch {epoch}/{self.epochs}', leave=False):
                Q1, Q2, QT1, QT2, actor = self.train_step(
                    Q1, 
                    Q2, 
                    QT1, 
                    QT2, 
                    actor, 
                    state, 
                    action, 
                    reward, 
                    state_, 
                    done
                )
        return Q1, Q2, QT1, QT2, actor

class Q_Model(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2):
        super().__init__()
        self.Q_net = nn.Sequential(
            nn.Linear(in_features=(state_size + action_size), out_features=hidden_size_1),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size_2, out_features=1)
        )
    def forward(self, s, a):
        x = torch.cat(
            [s,a],
            dim=-1
        )
        return self.Q_net(x)

class Actor:
    def __init__(self, state_size, action_size, hidden_size_1, hidden_size_2):
        super().__init__()
        self.action_dim = action_size
        self.policy_net = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=hidden_size_1),
            nn.SiLU(),
            nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
            nn.SiLU()
        )
        self.mu_head = nn.Linear(in_features=hidden_size_2, out_features=action_size)
        self.log_sig_head = nn.Linear(in_features=hidden_size_2, out_features=action_size)
        self.standard_normal = Normal(loc=0.0, scale=1.0)

    def forward(self, x):
        x = self.policy_net(x)
        mu = self.mu_head(x)
        log_sig = self.log_sig_head(x)
        sigma = torch.exp(log_sig)
        return mu, sigma
    
    def act(self, state):
        batch_size = state.shape[0]
        mu, sigma = self.forward(state)
        ep = self.standard_normal.sample(sample_shape=(batch_size, self.action_dim))
        log_prob_per_element = self.standard_normal.log_prob(ep)
        log_prob = log_prob_per_element.sum(dim=1)
        z = mu + sigma * ep
        return nn.Tanh(z), log_prob

def test(test_env, actor: Actor, tests, device):
    rewards = []
    for _ in tqdm(range(tests), desc='Testing Policy', leave=False):
        state, _ = test_env.reset(seed=50)
        done = False
        ep_reward = 0
        while not done:
            action, _ = actor.act(state)
            state_, reward, terminated, truncated, _ = test_env.step(action)
            done = (terminated or truncated)
            state = state_
            ep_reward += reward
        rewards.append(ep_reward)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    return torch.mean(reward_tensor)

def train():
    dataset = Dataset()
    Q_net_1 = Q_Model()
    Q_net_2 = Q_Model()
    target_Q1 = Q_Model()
    target_Q2 = Q_Model()
    actor = Actor()

    for iteration in range(iters):
        dataset.rollout(update_interval, env, actor)
        Q_net_1, Q_net_2, target_Q1, target_Q2, actor = dataset.update_networks(
            Q_net_1, 
            Q_net_2, 
            target_Q1, 
            target_Q2, 
            actor
        )  
        reward = test(test_env)     