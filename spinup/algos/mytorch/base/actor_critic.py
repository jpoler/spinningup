import math

import numpy as np
from gym.spaces import Box, Discrete

import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def kaiming_uniform(grouped_params):
    for i, (w, b) in enumerate(grouped_params):
        if i == len(grouped_params) - 1:
            torch.nn.init.uniform_(w.data, a=-3e-3, b=3e-3)
            torch.nn.init.uniform_(b.data, a=-3e-3, b=3e-3)
        else:
            torch.nn.init.kaiming_uniform(w.data, mode="fan_in", nonlinearity="relu")
            b.data.zero_()

def orthogonal_init(grouped_params):
    for i, (w, b) in enumerate(grouped_params):
        gain = math.sqrt(2.)
        print(w.shape)
        if i == len(grouped_params) - 1 and w.shape[0] == 1:
            gain = 1.
        elif i == len(grouped_params) - 1:
            gain = 0.01
        torch.nn.init.orthogonal_(w.data, gain=gain)
        b.data.zero_()


# TODO have the code detect an observation shape that requires convolution
# transparently and computes the appropriate sizes.
def mlp(sizes, activation, output_activation=torch.nn.Identity, conv=False, init=None):
    if not init:
        init = orthogonal_init

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

    seq = torch.nn.Sequential(*layers)

    params = list(seq.parameters())
    grouped_params = [params[i:i+2] for i in range(0, len(params), 2)]
    init(grouped_params)

    return seq


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class Actor(torch.nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, deterministic=False):
        if deterministic:
            return self._deterministic_action(obs)

        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, conv=False, init=None):
        super().__init__()
        self._mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, conv=conv, init=init)

    def _distribution(self, obs):
        logits = self._mlp(obs.float())
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation=torch.nn.Tanh, scale=None, init=None):
        super().__init__()
        self._loc_dim = torch.prod(torch.as_tensor(act_dim)).item()
        log_scale = np.log(scale) if scale else -0.5
        log_scale_tensor = log_scale * np.ones(self._loc_dim, dtype=np.float32)
        self._log_scale = torch.nn.Parameter(torch.as_tensor(log_scale_tensor), requires_grad=False)
        self._mlp = mlp(
            [obs_dim] + list(hidden_sizes) + [self._loc_dim],
            activation,
            output_activation=output_activation,
            init=init,
        )

    def _deterministic_action(self, obs):
        return self._mlp(obs.float())

    def _distribution(self, obs):
        loc = self._mlp(obs.float())
        return Normal(loc, torch.exp(self._log_scale))

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(torch.nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, act_dim=None, conv=False, init=None):
        super().__init__()
        if act_dim:
            self._mlp = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, conv=conv, init=init)
        else:
            self._mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation, conv=conv, init=init)

    def forward(self, obs, act=None):
        if act is not None:
            # TODO experiment with whether autodiff works through a view
            combined = torch.cat(
                (
                    torch.flatten(obs, start_dim=1),
                    torch.flatten(act, start_dim=1),
                ), 1)
            return torch.squeeze(self._mlp(combined), -1)
        return torch.squeeze(self._mlp(obs.float()), -1)

class MLPActorCritic(torch.nn.Module):


    def __init__(
            self,
            observation_space,
            action_space,
            hidden_sizes=(64,64),
            activation=torch.nn.Tanh,
            actor_gaussian_noise=None,
            conv=False,
            deterministic=False,
            q_net=False,
            init=None,
    ):
        super().__init__()

        self._obs_dim = observation_space.shape[0]
        self._q_net = q_net
        self._deterministic = deterministic
        self._action_low = torch.nn.Parameter(torch.as_tensor(action_space.low), requires_grad=False)
        self._action_high = torch.nn.Parameter(torch.as_tensor(action_space.high), requires_grad=False)
        if isinstance(action_space, Discrete):
            self._act_dim = action_space.n
            self.pi = MLPCategoricalActor(
                self._obs_dim,
                self._act_dim,
                hidden_sizes,
                activation,
                conv=conv,
                init=init,
            )
        elif isinstance(action_space, Box):
            self._act_dim = action_space.shape
            self.pi = MLPGaussianActor(
                self._obs_dim,
                self._act_dim,
                hidden_sizes,
                activation,
                scale=actor_gaussian_noise,
                init=init,
            )

        self.v = MLPCritic(self._obs_dim, hidden_sizes, activation, act_dim=action_space.shape[0] if q_net else None, conv=conv)

    def step(self, obs, device=None):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)[None, :]
        with torch.no_grad():
            dist = self.pi._distribution(obs)
            act = dist.sample().clip(min=self._action_low, max=self._action_high)
            logp = self.pi._log_prob_from_distribution(dist, act)
            val = self.v(obs, act=act if self._q_net else None)
        return act.squeeze(0).cpu().numpy(), val.squeeze(0).cpu().numpy(), logp.squeeze(0).cpu().numpy()

    def act(self, obs):
        with torch.no_grad():
            act = self.pi(obs, deterministic=True) if self._deterministic else self.pi._distribution(obs).sample()
            return act.clip(min=self._action_low, max=self._action_high).cpu().numpy()
