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
    def __init__(self):
        pass

    def linearise(self, )

    def plan(self, planning_horizon, dynamics_model, reward_model):
        pass



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
