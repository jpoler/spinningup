import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        print(sizes[i], sizes[i+1])
        layers += [nn.Linear(sizes[i], sizes[i+1]), act()]
    print(layers)
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self._mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self._mlp.forward(obs.float())
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self._loc_dim = torch.prod(torch.as_tensor(act_dim)).item()
        log_scale = -0.5 * np.ones(self._loc_dim, dtype=np.float32)
        self._log_scale = torch.nn.Parameter(torch.as_tensor(log_scale))
        self._mlp = mlp([obs_dim] + list(hidden_sizes) + [self._loc_dim], activation)

    def _distribution(self, obs):
        loc = self._mlp(obs.float())
        return Normal(loc, torch.exp(self._log_scale))

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self._mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return self._mlp.forward(obs.float())

class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        self._obs_dim = observation_space.shape[0]
        if isinstance(action_space, Discrete):
            self._act_dim = action_space.n
            self.pi = MLPCategoricalActor(self._obs_dim, self._act_dim, hidden_sizes, activation)
        elif isinstance(action_space, Box):
            self._act_dim = action_space.shape
            self.pi = MLPGaussianActor(self._obs_dim, self._act_dim, hidden_sizes, activation)

        self.v = MLPCritic(self._obs_dim, hidden_sizes, activation)

    def step(self, obs):
        obs = torch.from_numpy(obs)
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            act = dist.sample()
            logp = self.pi._log_prob_from_distribution(dist, act)
            val = self.v(obs)
        return act.numpy(), val.numpy(), logp.numpy()

    def act(self, obs):
        with torch.no_grad():
            return self.pi._distribution(obs).sample().numpy()
