import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm

class DataSet:
    def __init__(self, state_dim, action_dim, T, device):
        self.states = torch.zeros(size=(T, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros(size=(T, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(size=(T, 1), dtype=torch.float32, device=device)
        self.pred_values = torch.zeros(size=(T, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros(size=(T, 1), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size=(T, 1), dtype=torch.float32, device=device)
        self.t = 0
        self.T = T

        self.advantages = torch.zeros((T, 1), dtype=torch.float32, device=device)
        self.returns = torch.zeros((T, 1), dtype=torch.float32, device=device)
    
    def add(self, state, action, reward, pred_value, done, log_prob):
        self.states[self.t] = torch.tensor(state, dtype=torch.float32, device=self.states.device)
        self.actions[self.t] = torch.tensor(action, dtype=torch.float32, device=self.actions.device)
        self.rewards[self.t] = reward
        self.dones[self.t] = done
        self.log_probs[self.t] = log_prob
        self.values[self.t] = pred_value
        self.t += 1

    def compute_advantage_and_returns(self, critic, lambda_, gamma):
        adv_ = 0
        with torch.no_grad():
            value_ = critic.forward(self.state[self.t]) * (1 - self.dones[self.t])
            R = critic.forward(self.state[self.t])

        for i in reversed(range(self.t)):
            if self.dones[i]:
                value_ = 0
                R = 0
                TD_error = self.reward[i] + gamma * value_ - self.values[i]
                adv = TD_error
            else:
                TD_error = self.reward[i] + gamma * value_ - self.values[i]
                adv = TD_error + gamma * lambda_ * adv_ 
            adv_ = adv
            R = self.rewards[i] 
            value_ = self.values[i]
            return_ = self.rewards[i] + gamma * R


    def get_minibatches(self, batch_size):
        idxs = np.arange(self.t)
        np.random.shuffle(idxs)
        for start in range(0, self.t, batch_size):
            end = start + batch_size
            batch_idx = idxs[start:end]
            yield (
                self.states[batch_idx],
                self.actions[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
                self.log_probs[batch_idx],
                self.pred_values[batch_idx]
            )

    def reset(self):
        self.t = 0

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size_1, hidden_size_2):
        super.__init__()
        self.L1 = nn.Linear(state_size, hidden_size_1)
        self.L2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.L3 = nn.Linear(hidden_size_2, 1)
    
    def forward(self, x):
        x = self.L1(x)
        x = nn.ReLU(x)
        x = self.L2(x)
        x = nn.ReLU(x)
        return self.L3(x)

class Actor(nn.Module):
    def __init__(self, state_size, hidden_size_1, hidden_size_2, action_space_size):
        super.init()
        self.L1 = nn.Linear(state_size, hidden_size_1)
        self.L2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mu = nn.Linear(hidden_size_2, action_space_size)
        self.log_std = nn.Linear(hidden_size_2, action_space_size)

    def forward(self, x):
        x = self.L1(x)
        x = nn.ReLU(x)
        x = self.L2(x)
        x = nn.ReLU(x)
        mean = self.mu(x)
        std_dev = torch.exp(self.log_std(x))
        return mean, std_dev
    
    def get_dist(self, state):
        mean, std_dev = self.forward(state)
        return torch.distributions.Normal(loc=mean, scale=std_dev)

    def act(self, state):
        action_distribution = self.get_dist(state)
        action = action_distribution.rsample()
        log_prob = action_distribution.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

def ppo_update(actor, critic, dataset, actor_optimizer, critic_optimizer, batch_size, BETA, clip_eps=0.2):
    for states, actions, returns, advantages, old_log_probs, values in dataset.get_minibatches(batch_size):

        dist = actor.get_dist(states)
        new_log_probs = dist.log_prob(actions).sum(dims=-1, keepdim=True)
        ratios = (new_log_probs - old_log_probs).exp()

        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        actor_loss = actor_loss - BETA 

        value_loss = (critic(states) - returns).pow(2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

def policy_rollout(env, actor, critic, dataset, seed):
        state, _ = env.reset(seed)
        done = False
        for t in range(dataset.T):
            action, log_prob = actor.act(state)
            state_, reward, terminated, truncated, _ = env.step(action)
            done = (terminated or truncated)
            value = critic(state)
            dataset.add(
                state,
                action,
                reward,
                value,
                done,
                log_prob
            )
            state = state_
            if done:
                seed += 1
                state, _ = env.reset(seed)
                done = False

def main():
    if torch.cuda.is_available():
        device = 'cuda'
        print("cuda selected as device")
    else:
        device = 'cpu'
        print("cpu selected as device")

    env_id = 'LunarLander-V3'
    env = gym.make(env_id, continuous=True)

    seed = 43
    iterations = 500
    epochs = 20
    T = 600
    N = 1
    hs_c1 = 16
    hs_c2 = 16
    hs_a1 = 16
    hs_a2 = 16

    Learn_Rate_c = 0.001
    beta_1_c = 0.999
    beta_2_c = 0.9
    Learn_Rate_a = 0.001
    beta_1_a = 0.999
    beta_2_a = 0.9
    gamma = 0.999
    lambda_ = 0.8
    epsilon = 0.2
    batch_size = 50

    state_size = 8
    action_space_size = 4

    critic = Critic(
        state_size, 
        hs_c1,
        hs_c2
    )
    actor = Actor(
        state_size, 
        hs_a1, 
        hs_a2,
        action_space_size
    )

    optimiser_actor = torch.optim.Adam(
        actor.parameters(), 
        lr=Learn_Rate_a, 
        betas=(beta_1_a, beta_2_a)
    )
    optimiser_critic = torch.optim.Adam(
        critic.parameters(), 
        lr=Learn_Rate_c, 
        betas=(beta_1_c, beta_2_c)
    )

    dataset = DataSet(
        state_size, 
        action_space_size, 
        T, 
        device
    )

    for iteration in range(iterations):
        policy_rollout(
            env, 
            actor, 
            critic, 
            dataset, 
            seed
        )
        dataset.compute_advantage_and_returns(
            lambda_,
            gamma
        )
        for epoch in range(epochs):
            ppo_update(
                actor,
                critic,
                dataset,
                optimiser_actor,
                optimiser_critic,
                batch_size,
                epsilon
            )
                
if __name__ == "__main__":
    print("Running Training...")
    main()
    print("Training Complete")

