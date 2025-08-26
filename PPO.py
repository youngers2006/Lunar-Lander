import gymnasium as gym
import jax 
import jax.numpy as jnp
import jax.nn as nn
import optax
import flax.nnx as nnx
import distrax
import math

class Critic(nnx.Module):
    # critic takes state and return value
    def __init__(self, state_size, hidden_size, rngs):
        self.L1 = nnx.Linear(state_size, hidden_size, rngs)
        self.L2 = nnx.Linear(hidden_size, 1, rngs)
    
    def __call__(self, x):
        x = self.L1(x)
        x = nnx.relu(x)
        x = self.L2(x)
        return x

class Actor(nnx.Module):
    def __init__(self, state_size, hidden_size, action_space_size, rngs):
        self.L1 = nnx.Linear(state_size, hidden_size, rngs=rngs)
        self.L2 = nnx.Linear(hidden_size, action_space_size * 2, rngs=rngs)

    def __call__(self, x):
        x = self.L1(x)
        x = nnx.relu(x)
        x = self.L2(x)
        mean, log_std = jnp.split(
            x, 
            2, 
            axis=-1
        )
        std_dev = jnp.exp(log_std)
        return mean, std_dev

    def act(self, state, key):
        mean, std_dev = self(state)
        action_distribution = distrax.Normal(loc=mean, scale=std_dev) 
        action, log_prob = action_distribution.sample_and_log_prob(seed=key)
        return action, jnp.sum(log_prob, axis=-1)

def compute_advantages():
    pass

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

env_id = 'LunarLander-V3'
env = gym.make(env_id)

seed = 43
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

