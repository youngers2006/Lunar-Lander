import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class dynamics_model:
    def __init__(self, state_size, action_size, hidden1_size, hidden2_size):
        self.layer1 = nn.Linear(in_features=(state_size + action_size), out_features=hidden1_size)
        self.layer2 = nn.Linear(in_features=hidden1_size, out_features=hidden2_size)
        self.layer3 = nn.Linear(in_features=hidden2_size, out_features=(state_size))

    def forward(self, x):
        x = self.layer1(x)
        x = nn.SiLU(x)
        x = self.layer2(x)
        x = nn.SiLU(x)
        return self.layer3(x)

class reward_model:
    def __init__(self, state_size, action_size, hidden1_size, hidden2_size):
        self.layer1 = nn.Linear(in_features=(state_size + action_size), out_features=hidden1_size)
        self.layer2 = nn.Linear(in_features=hidden1_size, out_features=hidden2_size)
        self.layer3 = nn.Linear(in_features=hidden2_size, out_features=(1,))
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.SiLU(x)
        x = self.layer2(x)
        x = nn.SiLU(x)
        return self.layer3(x)

class MPC:
    def __init__(self, ):

class iLQR:
    def __init__(self, ):



def main():
