import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataSet:
    pass

class DynamicsModel:
    def __init__(self, state_size, action_size, hidden1_size, hidden2_size):
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

class MPC:
    def __init__(self, planner, planning_horizon):
        self.planning_algorithm = planner
        self.horizon = planning_horizon

    def act(self):
        self.planning_algorithm.plan(self.horizon)

class iLQR:
    def __init__(self, dynamics, reward_fn, state_dim, action_dim, iter_num,  device='cpu'):
        self.device = device
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_iterations = iter_num

    def update_reward_and_dynamics_models(self, dynamics, reward_fn):
        self.dynamics = dynamics
        self.reward_fn = reward_fn

    def cost_function(self, state, action):
        reward = self.reward_fn.forward(state, action)
        cost = -reward.squeeze(-1)
        return cost

    def derivatives(self, state, action):
        self.reward_fn.eval()
        self.dynamics.eval()

        concat_sa = torch.cat(
            tensors=[state, action],
            dim=-1
        )

        def f(concat_input): 
            x = concat_input[state.shape[0]:]
            u = concat_input[:state.shape[0]]
            return self.dynamics.next_state(x, u)
        
        def c(concat_input): 
            x = concat_input[state.shape[0]:]
            u = concat_input[:state.shape[0]]
            return self.cost_function(x, u)

        F = torch.autograd.functional.jacobian(
            func=f, 
            inputs=(concat_sa), 
            create_graph=False, 
            strict=False
        )
        c = torch.autograd.functional.jacobian(
            func=f, 
            inputs=(concat_sa), 
            create_graph=False, 
            strict=False
        )
        C = torch.autograd.functional.hessian(
            func=c, 
            inputs=(concat_sa), 
            create_graph=False, 
            strict=False
        )

        return F, C, c
        

    def backwards_pass(self, planning_horizon, Ft, ft, Ct, ct, CTerminal, cTerminal):
        k_gains = torch.zeros(planning_horizon, self.action_dim, device=self.device)
        K_gains = torch.zeros(planning_horizon, self.action_dim, self.state_dim, device=self.device)

        Vt = CTerminal
        vt = cTerminal
        for t in reversed(range(planning_horizon)):
            Qt = Ct + Ft.T @ Vt @ Ft
            qt = ct + Ft.T @ Vt @ ft + Ft.T @ vt

            Qxx = Qt[0,0]
            Quu = Qt[1,1]
            Qux = Qt[1,0]
            Qxu = Qt[0,1]
            qu = qt[1]
            qx = qt[0]

            Quu_inv = torch.linalg.inv(Quu)

            Kt = - Quu_inv @ Qux
            kt = - Quu_inv @ qu

            Vt = Qxx + Qxu @ Kt + Kt.T @ Qux + Kt.T @ Quu @ Kt
            vt = qx + Qxu @ kt +  Kt.T @ Qux + Kt.T @ Quu @ kt

        return k_gains, K_gains
    
    def _forward_pass(self, planning_horizon, nominal_states, nominal_actions, k_gains, K_gains, terminal_cost_function):
        alpha = 1.0

        nominal_cost = self.get_total_cost()
        for _ in range(10):
            new_states = torch.zeros_like(nominal_states)
            new_actions = torch.zeros_like(nominal_actions)
            new_states[0] = nominal_states[0]

            for t in range(planning_horizon):
                dx = new_states[t] - nominal_states[t]
                new_actions = K_gains[t] @ dx + alpha * k_gains[t] + nominal_actions[t]

                with torch.no_grad():
                    new_states[t+1] = self.dynamics.next_state(new_states[t], new_actions[t])
            
            new_cost = self.get_total_cost() # sort later

            if new_cost < nominal_cost:
                return new_states, new_actions
            
            alpha *= 0.5

        return nominal_states, nominal_actions
    
    def plan():

        

            

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
