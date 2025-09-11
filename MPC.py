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

class Dataset:
    def __init__(self, state_size, action_size, env, update_interval, dataset_length, random_rollouts, seed, device):
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        self.random_rollouts = random_rollouts
        self.update_interval = update_interval
        self.env = env
        self.rewards = deque(max_len=dataset_length)
        self.states = deque(max_len=dataset_length)
        self.actions = deque(max_len=dataset_length)
        self.next_states = deque(max_len=dataset_length)

        self.sample_count = 0
        self.device = device

    def random_policy(self):
        action = self.env.action_space.sample()
        return action

    def random_rollout(self):
        state, _ = self.env.reset(seed=self.seed)
        for t in range(self.random_rollouts):
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
                state, _ = self.env.reset
                terminated = False ; truncated = False

    def rollout(self, policy):
        state, _ = self.env.reset(seed=self.seed)
        for t in range(self.update_interval):
            action = policy() # need to finish this bit
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
            rew_batch = itertools.islice(self.rewards, i, stop_index)
            state_batch = itertools.islice(self.states, i, stop_index)
            next_state_batch = itertools.islice(self.next_states, i, stop_index)
            action_batch = itertools.islice(self.actions, i, stop_index)

            rb = torch.tensor(np.array(rew_batch), dtype=torch.float32, device=self.device)
            sb = torch.tensor(np.array(state_batch), dtype=torch.float32, device=self.device)
            nsb = torch.tensor(np.array(next_state_batch), dtype=torch.float32, device=self.device)
            ab = torch.tensor(np.array(action_batch), dtype=torch.float32, device=self.device)

            yield rb, sb, nsb, ab


    def train_dynamics_and_reward(self, dynamics_model, reward_model, dyn_optimiser, rew_optimiser, batch_size):
        dyn_loss_total = 0.0
        rew_loss_total = 0.0
        for reward_batch, state_batch, next_state_batch, action_batch in self.batch_samples(batch_size):
            batch_dyn_loss = torch.mean((dynamics_model(state_batch, action_batch) - next_state_batch).pow(2))
            batch_rew_loss = torch.mean((reward_model(state_batch, action_batch) - reward_batch).pow(2))

            dyn_optimiser.zero_grad()
            batch_dyn_loss.backward()
            dyn_optimiser.step()

            rew_optimiser.zero_grad()
            batch_rew_loss.backward()
            rew_optimiser.step()

            dyn_loss_total += batch_dyn_loss
            rew_loss_total += batch_rew_loss

        return dyn_loss_total, rew_loss_total
        

class DynamicsModel:
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
    
    def next_state(self, state, action):
        mu, sigma = self.forward(state, action)
        state_dist = torch.distributions.Normal(
            loc=mu, 
            scale=sigma
        )
        delta_state = state_dist.rsample()
        next_state_prediction = state + delta_state
        return next_state_prediction


class RewardModel:
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


class iLQR:
    def __init__(
            self, 
            dynamics, 
            reward_fn, 
            state_dim, 
            action_dim, 
            iter_num, 
            target_weights=[100,100,100,100,50,20,0,0], 
            device='cpu'
        ):
        self.device = device
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_iterations = iter_num
        self.lambda_ = 1.0
        self.Q_final = torch.diag(torch.tensor(target_weights))

    def increase_lambda(self, factor):
        self.lambda_ *= factor

    def decrease_lambda(self, factor):
        self.lambda_ /= factor

    def update_reward_and_dynamics_models(self, dynamics, reward_fn):
        self.dynamics = dynamics
        self.reward_fn = reward_fn

    def cost_function(self, state, action):
        reward = self.reward_fn.forward(state, action)
        cost = -reward.squeeze(-1)
        return cost
    
    def terminal_cost_function(self, state_T, goal_state):
        delta_state = state_T - goal_state
        cost_T = (delta_state).T @ self.Q_final @ (delta_state)
        return cost_T
    
    def get_total_cost(self, states, actions, goal_state):
        final_cost = self.terminal_cost_function(states[-1], goal_state)
        running_cost = 0
        planning_horizon = actions.shape[0]
        for i in range(planning_horizon):
            running_cost += self.cost_function(states[i], actions[i])

        total_cost = running_cost + final_cost
        return total_cost

    def derivatives(self, states, actions, goal_state):

        planning_horizon = actions.shape[0]
        
        C_list = torch.zeros(planning_horizon, self.state_dim + self.action_dim, self.state_dim + self.action_dim, device=self.device)
        c_list = torch.zeros(planning_horizon, self.state_dim + self.action_dim, device=self.device)

        def cost_wrapped(sa_input):
                x = sa_input[:self.state_dim] 
                u = sa_input[self.state_dim:] 
                return self.cost_function(x, u)
        
        for t in range(planning_horizon):
            state = states[t]
            action = actions[t]
            concat_sa = torch.cat(tensors=[state, action], dim=-1)

            c_list[t] = torch.autograd.functional.jacobian(func=cost_wrapped, inputs=concat_sa)
            C_list[t] = torch.autograd.functional.hessian(func=cost_wrapped, inputs=concat_sa)
        
        def terminal_cost_wrapped(state_T):
            return self.terminal_cost_function(state_T, goal_state)
        
        last_state = states[-1]
        cT = torch.autograd.functional.jacobian(func=terminal_cost_wrapped, inputs=last_state)
        CT = torch.autograd.functional.hessian(func=terminal_cost_wrapped, inputs=last_state)
        
        return C_list, c_list, CT, cT
    
    def linearise_dynamics(self, states, actions):
        planning_horizon = actions.shape[0] 
        
        F_list = torch.zeros(planning_horizon, self.state_dim, self.state_dim + self.action_dim, device=self.device)
        f_list = torch.zeros(planning_horizon, self.state_dim, device=self.device)

        def dynamics_wrapped(xu_input): 
                x = xu_input[:self.state_dim] 
                u = xu_input[self.state_dim:] 
                return self.dynamics.next_state(x, u)
        
        for t in range(planning_horizon):
            state = states[t]
            action = actions[t]
            xu = torch.cat(tensors=[state, action], dim=-1)
    
            with torch.no_grad():
                x_next = dynamics_wrapped(xu)
            
            Ft = torch.autograd.functional.jacobian(func=dynamics_wrapped, inputs=xu)
            f_list[t] = x_next - Ft @ xu
            F_list[t] = Ft

        return F_list, f_list

    def backwards_pass(self, planning_horizon, F, f, C, c, CTerminal, cTerminal):
        k_gains = torch.zeros(planning_horizon, self.action_dim, device=self.device)
        K_gains = torch.zeros(planning_horizon, self.action_dim, self.state_dim, device=self.device)

        Vt = CTerminal
        vt = cTerminal
        for t in reversed(range(planning_horizon)):
            Ct, ct, Ft, ft = C[t], c[t], F[t], f[t]

            Qt = Ct + Ft.T @ Vt @ Ft
            qt = ct + Ft.T @ Vt @ ft + Ft.T @ vt

            Qxx = Qt[0:self.state_dim,0:self.state_dim]
            Quu = Qt[self.state_dim:,self.state_dim:]
            Qux = Qt[self.state_dim:,0:self.state_dim]
            Qxu = Qt[0:self.state_dim,self.state_dim:]
            qu = qt[self.state_dim:]
            qx = qt[0:self.state_dim]

            Quu_reg = Quu + self.lambda_ * torch.eye(self.action_dim, device=self.device)

            Kt = - torch.linalg.solve(Quu_reg, Qux)
            kt = - torch.linalg.solve(Quu_reg, qu)

            K_gains[t] = Kt
            k_gains[t] = kt

            Vt = Qxx + Qxu @ Kt + Kt.T @ Qux + Kt.T @ Quu_reg @ Kt
            vt = qx + Qxu @ kt +  Kt.T @ qu + Kt.T @ Quu_reg @ kt

        return k_gains, K_gains
    
    def _forward_pass(self, planning_horizon, nominal_states, nominal_actions, k_gains, K_gains, goal_state):
        alpha = 1.0
        success = False

        nominal_cost = self.get_total_cost(nominal_states, nominal_actions, goal_state)
        for _ in range(10):
            new_states = torch.zeros_like(nominal_states)
            new_actions = torch.zeros_like(nominal_actions)
            new_states[0] = nominal_states[0]

            for t in range(planning_horizon):
                dx = new_states[t] - nominal_states[t]
                new_actions[t] = nominal_actions[t] + alpha * k_gains[t] + K_gains[t] @ dx

                with torch.no_grad():
                    new_states[t+1] = self.dynamics.next_state(new_states[t], new_actions[t])
            
            new_cost = self.get_total_cost(new_states, new_actions, goal_state) 

            if new_cost < nominal_cost:
                success = True
                return new_states, new_actions, success
            
            alpha *= 0.5

        return nominal_states, nominal_actions, success
    
    def initial_guess(self, x0, planning_horizon):
        nominal_states = torch.zeros(planning_horizon+1, self.state_dim, device=self.device)
        nominal_states[0] = x0
        nominal_actions = torch.zeros(planning_horizon, self.action_dim, device=self.device)

        with torch.no_grad():
            for t in range(planning_horizon):
                nominal_states[t+1] = self.dynamics.next_state(nominal_states[t], nominal_actions[t])

        return nominal_states, nominal_actions
    
    def plan(self, planning_horizon, state_I, goal_state):
        nominal_states, nominal_actions = self.initial_guess(state_I, planning_horizon)
        for i in range(self.num_iterations):
            F, f = self.linearise_dynamics(nominal_states, nominal_actions)
            C, c, CT, cT = self.derivatives(nominal_states, nominal_actions, goal_state)
            k_gains, K_gains = self.backwards_pass(
                planning_horizon, 
                F, 
                f, 
                C, 
                c, 
                CT, 
                cT
            )
            new_states, new_actions, success = self._forward_pass(
                planning_horizon, 
                nominal_states, 
                nominal_actions, 
                k_gains, 
                K_gains, 
                goal_state
            )
            if success:
                nominal_states = new_states
                nominal_actions = new_actions
                self.decrease_lambda(factor=1.5)
            else:
                self.increase_lambda(factor=5.0)
                if self.lambda_ > 1e6:
                    print(f"algorithm couldnt converge after iteration {i+1}.")
        return nominal_actions.detach()

        
class MPC:
    def __init__(self, planner: iLQR, planning_horizon):
        self.planning_algorithm = planner
        self.horizon = planning_horizon

    def act(self):
        actions = self.planning_algorithm.plan(self.horizon)
        return actions[0]
            
def main():
    if torch.cuda.is_available():
        device = 'cuda'
        print("cuda selected as device")
    else:
        device = 'cpu'
        print("cpu selected as device")

    env_id = 'LunarLanderContinuous-v3'
    env = gym.make(env_id)

    data_set = DataSet()

    dynamics_model = DynamicsModel()
    reward_model = RewardModel()

    iLQR_planner = iLQR()
    MPC_controller = MPC(
        planner=iLQR_planner,
    )



if __name__ == "__main__":
    print("running training")
    main()
    print("training complete")
