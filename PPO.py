import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm

class Critic(nn.Module):
    # critic takes state and return value
    def __init__(self, state_size, hidden_size, rngs):
        super.__init__()
        self.L1 = nn.Linear(state_size, hidden_size, rngs)
        self.L2 = nn.Linear(hidden_size, 1, rngs)
    
    def forward(self, x):
        x = self.L1(x)
        x = nn.ReLU(x)
        x = self.L2(x)
        return x

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

    def act(self, state, key):
        mean, std_dev = self.forward(state)
        action_distribution = torch.distributions.Normal(loc=mean, scale=std_dev) 
        action, log_prob = action_distribution.sample_and_log_prob(seed=key)
        return action, jnp.sum(log_prob, axis=-1)
    
class ModelState(nnx.Object):
    def __init__(self, actor_GD, actor_params, actor_state, critic_GD, critic_params, critic_state):
        self.actor_GD = actor_GD
        self.actor_params = actor_params
        self.actor_state = actor_state
        self.critic_GD = critic_GD
        self.critic_params = critic_params
        self.critic_state = critic_state

def compute_advantages():
    return advantages

def getImportanceRatio():
    return importance_ratio

def surrogate_obj(advantage, importance_ratio, Value_pred, Value_target, action_dist_std_dev,*, epsilon, C1, C2):
    L_clip = jnp.min(jnp.array([
        importance_ratio * advantage, 
        jnp.clip(importance_ratio, 1 - epsilon, 1 + epsilon) * advantage
        ]), 
        axis=-1
    )
    L_vf = jnp.power(Value_pred - Value_target, 2) 
    entropy = 0.5 * (jnp.log(2 * math.pi * (action_dist_std_dev ** 2) + 1))
    return L_clip - C1 * L_vf + C2 * entropy 

@jax.jit
def surrogate_optimisation_step(actor_GD, actor_params, actor_state, critic_GD, critic_params, critic_state, opt_state_actor, opt_state_critic):

    def surrogate_obj_call():
        return 0 
    
    return new_params_Critic, new_paramsActor, new_stateCritic, new_stateActor, new_opt_stateActor, new_opt_stateCritic

def main():

    def policy_rollout(T, seed, ):
        data_buffer = 
        state, info = env.reset(seed)
        terminated = False
        truncated = False
        for t in T:
            action = actor()
            state_, reward, terminated, truncated, info = env.step(action)

            state = state_
            if terminated or truncated:
                seed += 1
                state, info = env.reset(seed)
                terminated = False
                truncated = False

        return data_dict

    env_id = 'LunarLander-V3'
    env = gym.make(env_id, continuous=True)

    seed = 43
    iterations = 500
    epochs = 100
    T = 50
    N = 1
    hs_c = 16
    hs_a = 16
    Learn_Rate_c = 0.001
    beta_1_c = 0.999
    beta_2_c = 0.9
    Learn_Rate_a = 0.001
    beta_1_a = 0.999
    beta_2_a = 0.9

    state_size = 8
    action_space_size = 4

    base_key = jax.random.PRNGKey(seed)
    rngs = nnx.Rngs(base_key)

    critic = Critic(
        state_size, 
        hs_c, 
        rngs
    )

    actor = Actor(
        state_size, 
        hs_a, 
        action_space_size, 
        rngs
    )

    optimiser_critic = optax.adam(
        learning_rate=Learn_Rate_c, 
        b1=beta_1_c, 
        b2=beta_2_c
    )

    optimiser_actor = optax.adam(
        learning_rate=Learn_Rate_a, 
        b1=beta_1_a, 
        b2=beta_2_a
    )

    graph_defCritic, paramsCritic, stateCritic = nnx.split(critic, nnx.Param, nnx.State)
    graph_defActor, paramsActor, stateActor = nnx.split(critic, nnx.Param, nnx.State)

    opt_stateActor = optimiser_actor.init(paramsActor)
    opt_stateCritic = optimiser_critic.init(paramsCritic)

    current_state = ModelState(
        graph_defCritic, 
        paramsCritic, 
        stateCritic, 
        graph_defActor, 
        paramsActor, 
        stateActor, 
    )

    old_state = ModelState(
        graph_defCritic, 
        paramsCritic, 
        stateCritic, 
        graph_defActor, 
        paramsActor, 
        stateActor, 
    )

    for iteration in range(iterations):
        rollout_data = policy_rollout()
        advantages = compute_advantages()
        data_batches = batch_data()
        for epoch in range(epochs):
            for batch in tqdm(data_batches, desc=f'epoch {epoch}/{epochs} in iteration {iteration}/{iterations}', leave=False):
                importance_ratio = getImportanceRatio()
                new_params_Critic, new_paramsActor, new_stateCritic, new_stateActor, new_opt_stateActor, new_opt_stateCritic = surrogate_optimisation_step()
                current_state.actor_params = new_paramsActor
                current_state.actor_state = new_stateActor
                current_state.critic_params = new_params_Critic
                current_state.critic_state = new_stateCritic
                opt_stateActor = new_opt_stateActor
                opt_stateCritic = new_opt_stateCritic

        old_state.actor_params = current_state.actor_params 
        old_state.actor_state = current_state.actor_state
        old_state.critic_params = current_state.critic_params
        old_state.critic_state = current_state.critic_state

if __name__ == "__main__":
    print("Running Training...")
    main()
    print("Training Complete")

