import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
import itertools

class DataSet:
    def __init__(self, state_size, action_size, env, update_interval, dataset_length, seed, device):
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        self.update_interval = update_interval
        self.env = env
        self.rewards = deque(maxlen=dataset_length)
        self.states = deque(maxlen=dataset_length)
        self.actions = deque(maxlen=dataset_length)
        self.next_states = deque(maxlen=dataset_length)

        self.sample_count = 0
        self.device = device

    def random_policy(self):
        action = self.env.action_space.sample()
        return action

    def random_rollout(self, random_rollouts):
        state, _ = self.env.reset(seed=self.seed)
        for _ in tqdm(range(random_rollouts), desc="Rolling out random policy", leave=False):
            action = self.random_policy()
            state_, reward, terminated, truncated, _ = self.env.step(action)
            self.add_sample(
                reward, 
                state, 
                state_, 
                action
            )
            state = state_
            self.sample_count += 1
            if (terminated or truncated):
                state, _ = self.env.reset(seed=self.seed)

    def rollout(self, MPC):
        state, _ = self.env.reset(seed=self.seed)
        for _ in tqdm(range(self.update_interval), desc="rolling out MPC", leave=False):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = MPC.act(state_tensor)
            action = action.detach().cpu().numpy().flatten()
            state_, reward, terminated, truncated, _ = self.env.step(action)
            self.add_sample(
                reward, 
                state, 
                state_, 
                action
            )
            state = state_
            self.sample_count += 1
            if (terminated or truncated):
                state, _ = self.env.reset(seed=self.seed)
                terminated = False ; truncated = False

    def add_sample(self, reward, state, state_, action):
        self.rewards.append(reward)
        self.states.append(state)
        self.next_states.append(state_)
        self.actions.append(action)
    
    def batch_samples(self, batch_size):
        total_length = len(self.rewards)
        for i in range(0, total_length, batch_size):
            stop_index = min(i + batch_size, total_length)
            rew_batch = list(itertools.islice(self.rewards, i, stop_index))
            state_batch = list(itertools.islice(self.states, i, stop_index))
            next_state_batch = list(itertools.islice(self.next_states, i, stop_index))
            action_batch = list(itertools.islice(self.actions, i, stop_index))

            rb = torch.tensor(rew_batch, dtype=torch.float32, device=self.device)
            sb = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
            nsb = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=self.device)
            ab = torch.tensor(np.array(action_batch), dtype=torch.float32, device=self.device)
            yield rb, sb, nsb, ab

    def train_dynamics_and_reward(self, epochs, dynamics_model, reward_model, dyn_optimiser, rew_optimiser, batch_size):
        dynamics_model.to(self.device)
        reward_model.to(self.device)
        dyn_loss_total = []
        rew_loss_total = []
        for epoch in range(epochs):
            epoch_loss_dyn = 0.0
            epoch_loss_rew = 0.0
            batch_count = 0
            for reward_batch, state_batch, next_state_batch, action_batch in tqdm(self.batch_samples(batch_size), desc=f'Updating models. In epoch {epoch} / {epochs}', leave=False):
                batch_count += 1
                batch_dyn_loss = torch.mean((dynamics_model.next_state_det(state_batch, action_batch) - next_state_batch).pow(2))
                batch_rew_loss = torch.mean((reward_model(state_batch, action_batch) - reward_batch).pow(2))

                dyn_optimiser.zero_grad()
                batch_dyn_loss.backward()
                dyn_optimiser.step()

                rew_optimiser.zero_grad()
                batch_rew_loss.backward()
                rew_optimiser.step()

                epoch_loss_dyn += batch_dyn_loss
                epoch_loss_rew += batch_rew_loss

            dyn_loss_total.append(epoch_loss_dyn / batch_count)
            rew_loss_total.append(epoch_loss_rew / batch_count)

        return dyn_loss_total, rew_loss_total
        
class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size, hidden1_size, hidden2_size):
        super().__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(in_features=(state_size + action_size), out_features=hidden1_size),
             nn.SiLU(),
            nn.Linear(in_features=hidden1_size, out_features=hidden2_size),
            nn.SiLU()
        )
        self.mu_head = nn.Linear(in_features=hidden2_size, out_features=state_size)
        self.log_sigma_head = nn.Linear(in_features=hidden2_size, out_features=state_size)

    def forward(self, state, action):
        x = torch.cat(
            tensors=[state, action],
            dim=-1
        )
        x = self.shared_net(x)
        mu = self.mu_head(x)
        log_sigma = torch.clamp(
            input=self.log_sigma_head(x),
            max=2,
            min=-10
        )
        sigma = torch.exp(log_sigma)
        return mu, sigma
    
    def next_state_det(self, state, action):
        mu, _ = self.forward(state, action)
        delta_state = mu
        next_state_prediction = state + delta_state
        return next_state_prediction
    
    def next_state_stochastic(self, state, action):
        mu, sigma = self.forward(state, action)
        state_dist = torch.distributions.Normal(
                loc=mu, 
                scale=sigma
            )
        delta_state = state_dist.rsample()
        next_state_prediction = state + delta_state
        return next_state_prediction

class RewardModel(nn.Module):
    def __init__(self, state_size, action_size, hidden1_size, hidden2_size):
        super().__init__()
        self.reward_net = nn.Sequential(
            nn.Linear(in_features=(state_size + action_size), out_features=hidden1_size),
            nn.SiLU(),
            nn.Linear(in_features=hidden1_size, out_features=hidden2_size),
            nn.SiLU(),
            nn.Linear(in_features=hidden2_size, out_features=1)
        )

    def forward(self, state, action):
        x = torch.cat(
            tensors=[state, action],
            dim=-1
        )
        return self.reward_net(x)

class CEM: 
    def __init__(
            self, 
            num_samples, 
            horizon, 
            iterations, 
            num_elites, 
            reward_fn: RewardModel, 
            dynamics_fn: DynamicsModel, 
            action_dim, 
            alpha,
            init_mean=None, 
            init_var=None, 
            action_max=1.0, 
            action_min=-1.0, 
            device='cpu'
        ):
        self.num_samples = num_samples
        self.H = horizon
        self.iters = iterations
        self.N_elites = num_elites
        self.alpha = alpha
        self.reward_fn = reward_fn
        self.dynamics_fn = dynamics_fn
        self.device = device
        self.action_dim = action_dim
        self.var_floor = float(1e-4)
        self.action_max = action_max
        self.action_min = action_min

        if init_mean is None:
            self.mean = torch.zeros(horizon, action_dim, dtype=torch.float32, device=device)
        else:
            self.mean = torch.tensor(init_mean, dtype=torch.float32, device=device)
        
        if init_var is None:
            self.var = torch.full((horizon, action_dim), self.var_floor, dtype=torch.float32, device=device)
            self.init_var = self.var_floor
        else:
            self.var = torch.tensor(init_var, dtype=torch.float32, device=device)
            self.init_var = torch.clamp(init_var, min=self.var_floor)

    def update_models(self, reward_fn, dyn_fn):
        self.dynamics_fn = dyn_fn
        self.reward_fn = reward_fn

    def reset_mean_and_var(self):
        self.mean = torch.zeros_like(self.mean)
        self.var = torch.full_like(self.var, self.init_var)

    def sample_sequences(self):
        std = torch.sqrt(self.var.clamp(min=self.var_floor))
        z = torch.randn(self.num_samples, self.H, self.action_dim, device=self.device)
        samples = self.mean.unsqueeze(0) + z * std.unsqueeze(0)
        samples = torch.clamp(samples, min=self.action_min, max=self.action_max)
        return samples

    @torch.compile
    def evaluate_sequences(self, sequences, state_I):
        N, H, A = sequences.shape
        S = state_I.shape[0]
        s = state_I.expand(N, S)
        total_rewards = torch.zeros(N, dtype=torch.float32, device=self.device)

        for t in range(H):
            a = sequences[:, t, :]
            next_state = self.dynamics_fn.next_state_det(s,a)
            reward = self.reward_fn(s,a).view(-1)
            total_rewards += reward
            s = next_state
        return total_rewards

    def select_elites(self, sequences, evaluations):
        idx = torch.topk(evaluations.squeeze(-1), self.N_elites, dim=0).indices
        elites = sequences[idx]
        return elites
    
    def refit(self, elites):
        elite_mean = torch.mean(elites, dim=0)
        elite_var = torch.var(elites, dim=0, unbiased=False)
        new_mean = self.alpha * self.mean + (1.0 - self.alpha) * elite_mean
        new_var = torch.clamp(elite_var, min=self.var_floor)
        return new_mean, new_var
    
    def warmstart_fill_mean(self):
        return torch.zeros(self.action_dim, device=self.device)

    def warmstart_fill_var(self):
        return torch.ones_like(self.var[-1]) * self.init_var
    
    def warm_start_prep(self):
        self.mean = torch.roll(self.mean, shifts=-1, dims=0)
        self.mean[-1] = self.warmstart_fill_mean()
        self.var = torch.roll(self.var, shifts=-1, dims=0)
        self.var[-1] = self.warmstart_fill_var()

    def plan(self, state_I):
        state_I = torch.tensor(state_I, dtype=torch.float32, device=self.device)
        for _ in range(self.iters):
            sequences = self.sample_sequences()
            evals = self.evaluate_sequences(sequences, state_I)
            elites = self.select_elites(sequences, evals)
            new_mean, new_var = self.refit(elites)
            self.mean = new_mean
            self.var = new_var
        best_sequence = self.mean.detach().clone()
        self.warm_start_prep()
        return best_sequence

class MPC:
    def __init__(self, planner: CEM):
        self.planning_algorithm = planner

    def act(self, state):
        actions = self.planning_algorithm.plan(state)
        return actions[0]
            
def train():
    if torch.cuda.is_available():
        device = 'cuda'
        print("cuda selected as device")
    else:
        device = 'cpu'
        print("cpu selected as device")

    seed = 42
    env_id = 'LunarLanderContinuous-v3'
    env = gym.make(env_id)

    horizon = 25
    CEM_iters = 3
    CEM_elite_num = 50
    CEM_samples = 500
    alpha = 0.15
    update_interval = 10000
    random_rollouts = 200000
    dataset_length = 200000
    training_rollouts = 50
    hd1 = 128 ; hd2 = 64
    hr1 = 128 ; hr2 = 64
    epochs = 10
    batch_size = 1000
    learn_rate_rew = 0.001
    learn_rate_dyn = 0.001
    a_max = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
    a_min = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    data_set = DataSet(
        state_size, 
        action_size, 
        env, 
        update_interval, 
        dataset_length, 
        seed, 
        device
    )
    dynamics_model = DynamicsModel(
        state_size, 
        action_size, 
        hd1, 
        hd2
    ).to(device)
    reward_model = RewardModel(
        state_size, 
        action_size, 
        hr1, 
        hr2
    ).to(device)

    CEM_planner = CEM(
        num_samples=CEM_samples,
        horizon=horizon,
        iterations=CEM_iters,
        num_elites=CEM_elite_num,
        reward_fn=reward_model,
        dynamics_fn=dynamics_model,
        action_dim=action_size,
        alpha=alpha,
        action_max=a_max,
        action_min=a_min,
        device=device
    )
    MPC_controller = MPC(
        planner=CEM_planner
    )

    dynamics_optimiser = torch.optim.Adam(
        dynamics_model.parameters(),
        learn_rate_dyn
    )
    reward_optimiser = torch.optim.Adam(
        reward_model.parameters(),
        learn_rate_rew
    )

    progress_tracker = []
    print("started random rollouts")
    for _ in range(10):
        data_set.random_rollout(random_rollouts)
        data_set.train_dynamics_and_reward(
        epochs, 
        dynamics_model, 
        reward_model, 
        dynamics_optimiser, 
        reward_optimiser, 
        batch_size
    )
        MPC_controller.planning_algorithm.update_models(reward_model, dynamics_model)
    print("starting MPC rollouts")
    for _ in tqdm(range(training_rollouts), leave=False):
        data_set.rollout(MPC_controller)
        data_set.train_dynamics_and_reward(
            epochs, 
            dynamics_model, 
            reward_model, 
            dynamics_optimiser, 
            reward_optimiser, 
            batch_size
        )
        MPC_controller.planning_algorithm.update_models(reward_model, dynamics_model)
        reward_test = test(MPC_controller)
        progress_tracker.append(reward_test)

    return MPC_controller, progress_tracker

def test(MPC_controller, **kwargs):
    env_id = 'LunarLanderContinuous-v3'
    test_env = gym.make(env_id, **kwargs)
    test_runs = 3

    def test_rollout(MPC, env, test_runs, seed=42):
        reward_list = []
        for run in range(test_runs):
            state, _ = env.reset(seed=seed)
            terminated = False ; truncated = False
            run_reward = 0.0
            while not (terminated or truncated):
                action = MPC.act(state)
                action = action.detach().cpu().numpy().flatten()
                state_, reward, terminated, truncated, _ = env.step(action)
                run_reward += reward
                state = state_
            reward_list.append(run_reward)
        return reward_list
    
    rewards = test_rollout(MPC_controller, test_env, test_runs)
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    return mean_reward
            
def main():
    MPC_controller, progress = train()
    print("Finished training, beginning testing")
    rewards_test = test(MPC_controller, render_mode='human')
    print("Finished testing")
    plt.plot(progress)
    print(f'Trained controller scored a mean final reward of {rewards_test} having been trained')
    plt.show()

if __name__ == "__main__":
    print("running training")
    main()
    print("training complete")
