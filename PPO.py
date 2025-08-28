import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        self.pred_values[self.t] = pred_value
        self.t += 1

    def compute_advantage_and_returns(self, critic, lambda_, gamma):
        if self.t == 0:
            return

        with torch.no_grad():
            last_idx = self.t - 1
            last_done = self.dones[last_idx]  
            last_state = self.states[last_idx].unsqueeze(0)  
            last_value = critic(last_state).detach() * (1.0 - last_done)
            next_value = last_value
            R = last_value

        adv = 0.0
        for i in reversed(range(self.t)):
            mask = 1.0 - self.dones[i]
            delta = self.rewards[i] + gamma * next_value * mask - self.pred_values[i]
            adv = delta + gamma * lambda_ * mask * adv
            self.advantages[i] = adv
            R = self.rewards[i] + gamma * R * mask
            self.returns[i] = R
            next_value = self.pred_values[i]


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
        super().__init__()
        self.L1 = nn.Linear(state_size, hidden_size_1)
        self.L2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.L3 = nn.Linear(hidden_size_2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        return self.L3(x)

class Actor(nn.Module):
    def __init__(self, state_size, hidden_size_1, hidden_size_2, action_space_size, action_low, action_high, device):
        super().__init__()
        self.L1 = nn.Linear(state_size, hidden_size_1)
        self.L2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.mu = nn.Linear(hidden_size_2, action_space_size)
        self.log_std = nn.Linear(hidden_size_2, action_space_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)

    def forward(self, x):
        x = self.L1(x)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        mean = self.tanh(self.mu(x))
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        std_dev = torch.exp(log_std)
        return mean, std_dev
    
    def get_dist(self, state):
        mean, std_dev = self.forward(state)
        return torch.distributions.Normal(loc=mean, scale=std_dev)

    def act(self, state):
        action_distribution = self.get_dist(state)
        action = action_distribution.rsample()
        log_prob = action_distribution.log_prob(action).sum(dim=-1, keepdim=True)
        action = self.scale_action(action)
        return action, log_prob
    
    def scale_action(self, action):
        return self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)

def ppo_update(actor, critic, dataset, actor_optimizer, critic_optimizer, batch_size, BETA, clip_eps=0.2):
    actor_loss_total = 0
    value_loss_total = 0
    for states, actions, returns, advantages, old_log_probs, values in dataset.get_minibatches(batch_size):

        dist = actor.get_dist(states)
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        ratios = (new_log_probs - old_log_probs).exp()

        surrogate1 = ratios * advantages
        surrogate2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
        actor_loss = -torch.min(surrogate1, surrogate2).mean()
        entropy = dist.entropy().sum(dim=-1, keepdim=True).mean()
        actor_loss = actor_loss - BETA * entropy.mean()

        value_loss = (critic(states) - returns).pow(2).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        actor_loss_total += actor_loss
        value_loss_total += value_loss

    return actor_loss_total, value_loss_total

def policy_rollout(env, actor, critic, dataset, seed, device):
        state, _ = env.reset(seed=seed)
        done = False
        for t in range(dataset.T):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob = actor.act(state_tensor)
            action_np = action.detach().cpu().numpy().flatten()
            state_, reward, terminated, truncated, _ = env.step(action_np)
            done = (terminated or truncated)
            value = critic(state_tensor)
            dataset.add(
                torch.tensor(state, dtype=torch.float32, device=device),
                action.squeeze(0),
                torch.tensor([reward], dtype=torch.float32, device=device),
                value.detach(),
                torch.tensor([done], dtype=torch.float32, device=device),
                log_prob.detach()
            )
            state = state_
            if done:
                seed += 1
                state, _ = env.reset(seed=seed)
                done = False

def main():
    if torch.cuda.is_available():
        device = 'cuda'
        print("cuda selected as device")
    else:
        device = 'cpu'
        print("cpu selected as device")

    env_id = 'LunarLanderContinuous-v3'
    env = gym.make(env_id)

    seed = 43
    iterations = 500
    epochs = 10
    T = 1000
    N = 1
    hs_c1 = 16 ; hs_c2 = 8
    hs_a1 = 16 ; hs_a2 = 8

    Learn_Rate_c = 0.0001 ; Learn_Rate_a = 0.00001
    beta_1_c = 0.999 ; beta_2_c = 0.9
    beta_1_a = 0.999 ; beta_2_a = 0.9
    gamma = 0.999
    lambda_ = 0.8
    epsilon = 0.2
    batch_size = 50
    BETA = 0.01

    state_size = env.observation_space.shape[0]
    action_space_size = env.action_space.shape[0]

    critic = Critic(
        state_size, 
        hs_c1,
        hs_c2
    ).to(device)

    action_low = env.action_space.low
    action_high = env.action_space.high

    actor = Actor(
        state_size, 
        hs_a1, 
        hs_a2,
        action_space_size,
        action_low, 
        action_high,
        device
    ).to(device)

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

    actor_loss_iter = np.empty(shape=(iterations,1), dtype=np.float32)
    critic_loss_iter = np.empty(shape=(iterations,1), dtype=np.float32)
    
    for iteration in tqdm(range(iterations), leave=True):
        actor_loss_epochs = np.empty(shape=(epochs,1), dtype=np.float32)
        critic_loss_epochs = np.empty(shape=(epochs,1), dtype=np.float32)
        dataset.reset()
        policy_rollout(
            env, 
            actor, 
            critic, 
            dataset, 
            seed,
            device
        )
        dataset.compute_advantage_and_returns(
            critic,
            lambda_,
            gamma
        )
        for epoch in range(epochs):
            actor_loss_epoch, critic_loss_epoch = ppo_update(
                actor,
                critic,
                dataset,
                optimiser_actor,
                optimiser_critic,
                batch_size,
                BETA,
                epsilon
            )
            actor_loss_epochs[epoch] = actor_loss_epoch
            critic_loss_epochs[epoch] = critic_loss_epoch
        
        actor_loss_iter[iteration] = actor_loss_epochs.mean()
        critic_loss_iter[iteration] = critic_loss_epochs.mean()

    plt.plot(actor_loss_iter)
    plt.plot(critic_loss_iter)
    plt.show()




                
if __name__ == "__main__":
    print("Running Training...")
    main()
    print("Training Complete")

