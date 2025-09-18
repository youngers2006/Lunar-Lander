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
    def __init__(self, buffer_size, batch_size, train_epochs, Q1_optimiser, Q2_optimiser, actor_optimiser, alpha, tau, gamma, seed, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.epochs = train_epochs
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.seed = seed
        self.sample_count = 0
        self.opt_Q1 = Q1_optimiser
        self.opt_Q2 = Q2_optimiser
        self.opt_actor = actor_optimiser
        self.device = device

    def rollout(self, num_rollouts, env, actor):
        state, _ = env.reset(seed=self.seed)
        for _ in tqdm(range(num_rollouts), desc=f'Running Rollouts', leave=False):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, _ = actor.act(state_tensor)
            action = action.cpu().numpy().squeeze(0)
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
        permuted_idx = np.random.permutation(total_length)
        for i in range(0, total_length, self.batch_size):
            stop_idx = i + self.batch_size
            batch_idx = permuted_idx[i:stop_idx]

            if batch_idx.shape[0] < self.batch_size:
                continue

            batch = list(self.buffer[i] for i in batch_idx)

            states, actions, rewards, next_states, dones = zip(*batch)

            rb = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device).unsqueeze(1)
            sb = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            nsb = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            ab = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
            db = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device).unsqueeze(1)
            yield sb, ab, rb, nsb, db

    def soft_update(self, target, main):
        with torch.no_grad():
            for target_param, main_param in zip(target.parameters(), main.parameters()):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
        return target
    
    def train_step(self, Q1, Q2, QT1, QT2, actor, state, action, reward, state_, done):
        with torch.no_grad():
            action_, log_prob_ = actor.act(state_)
            Q_pred_targ1 = QT1.forward(state_, action_)
            Q_pred_targ2 = QT2.forward(state_, action_)
            min_Q_targ = torch.min(Q_pred_targ1, Q_pred_targ2)
            y = reward + self.gamma * (1 - done) * (min_Q_targ - self.alpha * log_prob_)

        Q_pred1 = Q1.forward(state, action)
        Q_pred2 = Q2.forward(state, action)

        loss_Q_1 = nn.functional.mse_loss(Q_pred1, y)
        loss_Q_2 = nn.functional.mse_loss(Q_pred2, y)

        self.opt_Q1.zero_grad()
        loss_Q_1.backward()
        self.opt_Q1.step()

        self.opt_Q2.zero_grad()
        loss_Q_2.backward()
        self.opt_Q2.step()

        action_pred, log_prob = actor.act(state)
        Q_pred_1_ = Q1.forward(state, action_pred)
        Q_pred_2_ = Q2.forward(state, action_pred)
        min_Q = torch.min(Q_pred_1_, Q_pred_2_)

        loss_actor = (self.alpha * log_prob - min_Q).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
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

class Actor(nn.Module):
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

    def forward(self, x):
        x = self.policy_net(x)
        mu = self.mu_head(x)
        log_sig = torch.clamp(self.log_sig_head(x), min=-20, max=2)
        sigma = torch.exp(log_sig)
        return mu, sigma
    
    def act(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma)
        z = dist.rsample() 
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

def test(test_env, actor: Actor, tests, device):
    rewards = []
    for _ in tqdm(range(tests), desc='Testing Policy', leave=False):
        state, _ = test_env.reset(seed=50)
        done = False
        ep_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, _ = actor.act(state_tensor)
            action.cpu().numpy().squeeze(0)
            state_, reward, terminated, truncated, _ = test_env.step(action)
            done = (terminated or truncated)
            state = state_
            ep_reward += reward
        rewards.append(ep_reward)
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    return torch.mean(reward_tensor)

def train():
    if torch.cuda.is_available():
        device = 'cuda'
        print("cuda selected as device")
    else:
        device = 'cpu'
        print("cpu selected as device")

    env_id = 'LunarLanderContinuous-v3'
    env = gym.make(env_id)
    test_env_1 = gym.make(env_id)
    test_env_2 = gym.make(env_id, render_mode='human')
    seed = 43

    iterations = 500 
    epochs = 10
    update_interval = 10000
    train_tests = 3
    buffer_size = 500000
    batch_size = 50
    
    h1Q = 64 ; h2Q = 32
    h1A = 64 ; h2A = 32

    Learn_Rate_Q = 3e-4 ; Learn_Rate_a = 3e-4
    beta_1_Q = 0.999 ; beta_2_Q = 0.9
    beta_1_a = 0.999 ; beta_2_a = 0.9
    gamma = 0.999
    tau = 0.995
    alpha = 0.15
    batch_size = 50

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    Q_net_1 = Q_Model(state_size, action_size, h1Q, h2Q)
    Q_net_2 = Q_Model(state_size, action_size, h1Q, h2Q)
    target_Q1 = Q_Model(state_size, action_size, h1Q, h2Q)
    target_Q2 = Q_Model(state_size, action_size, h1Q, h2Q)
    actor = Actor(state_size, action_size, h1A, h2A)

    Q_net_1.to(device) ; Q_net_2.to(device) ; target_Q1.to(device) ;  target_Q2.to(device) ;  actor.to(device)

    Q1_optimiser = torch.optim.Adam(Q_net_1.parameters(), lr=Learn_Rate_Q, betas=(beta_1_Q, beta_2_Q))
    Q2_optimiser = torch.optim.Adam(Q_net_2.parameters(), lr=Learn_Rate_Q, betas=(beta_1_Q, beta_2_Q))
    actor_optimiser = torch.optim.Adam(actor.parameters(),  lr=Learn_Rate_a, betas=(beta_1_a, beta_2_a))

    dataset = Dataset(
        buffer_size, 
        batch_size, 
        epochs, 
        Q1_optimiser, 
        Q2_optimiser, 
        actor_optimiser, 
        alpha,
        tau, 
        gamma, 
        seed, 
        device
    )
    rewards = []
    for iteration in range(iterations):
        dataset.rollout(update_interval, env, actor)
        Q_net_1, Q_net_2, target_Q1, target_Q2, actor = dataset.update_networks(
            Q_net_1, 
            Q_net_2, 
            target_Q1, 
            target_Q2, 
            actor
        )  
        test_reward = test(test_env_1, actor, train_tests, device)
        rewards.append(test_reward)
    test_reward_final = test(test_env_2, actor, 50, device)
    return actor, test_reward, test_reward_final

if __name__ == "__main__":
    print("Running Training")
    actor, test_reward, trained_reward = train()
    print("Training Complete")