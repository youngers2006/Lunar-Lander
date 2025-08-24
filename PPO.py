import gymnasium as gym
import jax 
import jax.numpy as jnp
import jax.nn as nn
import optax
import flax.nnx as nnx

env_id = 'Lunar-LanderV2'
env = gym.make(env_id)

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
        self.L2 = nnx.Linear(hidden_size, action_space_size, rngs=rngs)

    def __call__(self, x):
        x = self.L1(x)
        x = nnx.relu(x)
        x = self.L2(x)
        return x

    def act(self, state, key):
        logits = self(state) # probability of each action
        action = jax.random.categorical(key=key, logits=logits)
        probabilities = nnx.softmax(logits, axis=-1)
        log_prob = jnp.log(probabilities[action])
        return action

