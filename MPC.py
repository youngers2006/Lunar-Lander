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


class iLQR:
    def __init__(self, dynamics, reward_fn, state_dim, action_dim, iter_num, device='cpu'):
        self.device = device
        self.dynamics = dynamics
        self.reward_fn = reward_fn
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_iterations = iter_num
        self.lambda_ = 1e-6

    def increase_lambda(self, factor):
        self.lambda_ *= factor

    def update_reward_and_dynamics_models(self, dynamics, reward_fn):
        self.dynamics = dynamics
        self.reward_fn = reward_fn

    def cost_function(self, state, action):
        reward = self.reward_fn.forward(state, action)
        cost = -reward.squeeze(-1)
        return cost
    
    def terminal_cost_function(self, state_T):
        
        return cost_T
    
    def get_total_cost(self, states, actions):

        return cost

    def derivatives(self, states, actions):

        def deriv_single(state, action):
            self.reward_fn.eval()
            concat_sa = torch.cat(
                tensors=[state, action],
                dim=-1
            )
        
            def c(concat_input): 
                x = concat_input[state.shape[0]:]
                u = concat_input[:state.shape[0]]
                return self.cost_function(x, u)

            c = torch.autograd.functional.jacobian(
                func=c, 
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
            return C, c
        
        T = states.shape[0]
        Ct_list = torch.zeros(T, self.state_dim, self.state_dim + self.action_dim, device=self.device)
        ct_list = torch.zeros(T, self.state_dim, device=self.device)

        for t in range(T-1):
            Ct, ct = deriv_single(states[t], actions[t])
            Ct_list[t] = Ct
            ct_list[t] = ct

        CT = torch.autograd.functional.hessian(
            func=self.terminal_cost_function,
            inputs=(states[T]),
            create_graph=False,
            strict=False
        )
        cT = torch.autograd.functional.jacobian(
            func=self.terminal_cost_function,
            inputs=(states[T]),
            create_graph=False,
            strict=False
        )
        
        return Ct_list, ct_list, CT, cT
    
    def linearise_dynamics(self, states, actions):
        T = states.shape[0]
        Ft_list = torch.zeros(
            T, 
            self.state_dim, 
            self.state_dim + self.action_dim, 
            device=self.device
        )
        ft_list = torch.zeros(
            T, 
            self.state_dim, 
            device=self.device
        )

        def lin_single(x,u):
            self.dynamics.eval()
            x = x.detach().clone().requires_grad_(True)
            u = u.detach().clone().requires_grad_(True)
            xu = torch.cat(tensors=[x,u], dim=-1)

            def f(concat_input): 
                x = concat_input[x.shape[0]:]
                u = concat_input[:x.shape[0]]
                return self.dynamics.next_state(x, u)
        
            x_next = f(xu)
            Ft = torch.autograd.functional.jacobian(
                func=f, 
                inputs=xu, 
                create_graph=False, 
                strict=False
            )
            ft = x_next.detach() - Ft @ xu.detach()
            return Ft, ft
        
        for t in range(T):
            Ft, ft = lin_single(states[t], actions[t])
            Ft_list[t] = Ft
            ft_list[t] = ft

        return Ft_list, ft_list

    def backwards_pass(self, planning_horizon, Ft, ft, Ct, ct, CTerminal, cTerminal):
        k_gains = torch.zeros(planning_horizon, self.action_dim, device=self.device)
        K_gains = torch.zeros(planning_horizon, self.action_dim, self.state_dim, device=self.device)

        Vt = CTerminal
        vt = cTerminal
        for t in reversed(range(planning_horizon)):
            Qt = Ct + Ft[t].T @ Vt @ Ft[t]
            qt = ct + Ft.T @ Vt @ ft[t] + Ft[t].T @ vt

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
    
    def _forward_pass(self, planning_horizon, nominal_states, nominal_actions, k_gains, K_gains):
        alpha = 1.0
        success = False

        nominal_cost = self.get_total_cost(nominal_states, nominal_actions)
        for _ in range(10):
            new_states = torch.zeros_like(nominal_states)
            new_actions = torch.zeros_like(nominal_actions)
            new_states[0] = nominal_states[0]

            for t in range(planning_horizon):
                dx = new_states[t] - nominal_states[t]
                new_actions[t] = nominal_actions[t] + alpha * k_gains[t] + K_gains[t] @ dx

                with torch.no_grad():
                    new_states[t+1] = self.dynamics.next_state(new_states[t], new_actions[t])
            
            new_cost = self.get_total_cost(new_states, new_actions) 

            if new_cost < nominal_cost:
                success = True
                return new_states, new_actions, success
            
            alpha *= 0.5

        return nominal_states, nominal_actions, success
    
    def initial_guess(self, x0, planning_horizon):
        nominal_states = torch.zeros(planning_horizon+1, self.state_dim, device=self.device)
        nominal_states[0] = x0
        nominal_actions = torch.zeros(planning_horizon, self.action_dim, device=self.device)

        for t in range(planning_horizon):
            nominal_states[t+1] = self.dynamics.next_state(nominal_states[t], nominal_actions[t])

        return nominal_states, nominal_actions
    
    def plan(self, planning_horizon, state_I):
        nominal_states, nominal_actions = self.initial_guess(state_I, planning_horizon)
        for _ in range(self.num_iterations):
            Ft, ft = self.linearise_dynamics(nominal_states, nominal_actions)
            Ct, ct, CT, cT = self.derivatives(nominal_states, nominal_actions)
            k_gains, K_gains = self.backwards_pass(planning_horizon, Ft, ft, Ct, ct, CT, cT)
            new_states, new_actions, success = self._forward_pass(planning_horizon, nominal_states, nominal_actions, k_gains, K_gains)
            if success:
                nominal_states = new_states
                nominal_actions = new_actions
            else:
                self.increase_lambda(factor = 1.05)
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
