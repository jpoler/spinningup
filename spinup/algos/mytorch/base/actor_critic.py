import math

import numpy as np
from gym.spaces import Box, Discrete

import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# TODO break conv out into separate logic and integrate at a higher level
def mlp(sizes, activation, output_activation=torch.nn.Identity, conv=False):
    layers = []
    if conv:
        activation = torch.nn.ReLU
        layers += [torch.nn.Conv2d(4, 16, (8, 8), stride=4), activation()]
        layers += [torch.nn.Conv2d(16, 32, (4, 4), stride=2), activation()]
        layers += [torch.nn.Flatten()]
        # TODO remove hard coding
        sizes[0] = 9 * 9 * 32

    for i in range(len(sizes) - 1):
        act = output_activation if i == len(sizes) - 2 else activation
        print(sizes[i], sizes[i+1])
        layers += [torch.nn.Linear(sizes[i], sizes[i+1]), act()]
    print(layers)
    seq = torch.nn.Sequential(*layers)
    params = list(seq.parameters())
    grouped_params = [params[i:i+2] for i in range(0, len(params), 2)]
    for i, (w, b) in enumerate(grouped_params):
        gain = math.sqrt(2.)
        if i == len(grouped_params) - 1 and sizes[-1] == 1:
            gain = 1.
        elif i == len(grouped_params) - 1:
            gain = 0.01
        torch.nn.init.orthogonal_(w.data, gain=gain)
        b.data.zero_()
    return seq


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(torch.nn.Module):

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

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, conv=False):
        super().__init__()
        self._mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, conv=conv)

    def _distribution(self, obs):
        logits = self._mlp(obs.float())
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

    def deterministic_action(self, obs):
        return self._mlp(obs.float())

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(torch.nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, act_dim=None, conv=False):
        super().__init__()
        if act_dim:
            self._obs_mlp = mlp([obs_dim, hidden_sizes[0]], activation, output_activation=activation)
            self._act_mlp = mlp([act_dim, hidden_sizes[0]], activation, output_activation=activation)
            self._mlp = mlp([2*hidden_sizes[0]] + list(hidden_sizes) + [1], activation)
        else:
            self._mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation, conv=conv)

    def forward(self, obs, act=None):
        if act:
            obs_hidden = self._obs_mlp(obs.float())
            act_hidden = self._act_mlp(act.float())
            # TODO experiment with whether autodiff works through a view
            combined = torch.cat((torch.flatten(obs_hidden), torch.flatten(act_hidden)), 0)
            # TODO check whether torch.squeeze is actually needed
            return self._mlp(combined)
        return torch.squeeze(self._mlp(obs.float()), -1)

class MLPActorCritic(torch.nn.Module):


    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=torch.nn.Tanh, conv=False, deterministic=False, q_net=False):
        super().__init__()

        self._obs_dim = observation_space.shape[0]
        self._q_net = q_net
        self._deterministic = deterministic
        if isinstance(action_space, Discrete):
            self._act_dim = action_space.n
            self.pi = MLPCategoricalActor(self._obs_dim, self._act_dim, hidden_sizes, activation, conv=conv)
        elif isinstance(action_space, Box):
            self._act_dim = action_space.shape
            self.pi = MLPGaussianActor(self._obs_dim, self._act_dim, hidden_sizes, activation)

        self.v = MLPCritic(self._obs_dim, hidden_sizes, activation, act_dim=self._act_dim if q_net else None, conv=conv)

    def step(self, obs, device=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            act = dist.sample()
            logp = self.pi._log_prob_from_distribution(dist, act)
            val = self.v(obs, act=act if self._q_net else None)
        return act.cpu().numpy(), val.cpu().numpy(), logp.cpu().numpy()

    def act(self, obs):
        with torch.no_grad():
            act = self.pi.deterministic_action(obs) if self._deterministic else self.pi._distribution(obs).sample()
            return act.cpu().numpy()
