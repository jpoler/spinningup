from functools import partial
import time

from torch.optim import Adam
import numpy as np
import torch
import gym

from spinup.algos.mytorch.base.actor_critic import MLPActorCritic
from spinup.algos.mytorch.base.algorithm import Algorithm
from spinup.algos.mytorch.base.buffer import GAEBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class VPGAlgorithm(Algorithm):
    def __init__(
            self,
            env_fn,
            actor_critic=MLPActorCritic,
            ac_kwargs=None,
            gamma=0.99,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_v_iters=80,
            lam=0.97,
            **kwargs,
    ):
        buf_fn = partial(GAEBuffer, gamma=gamma, lam=lam)
        super().__init__(env_fn, buf_fn, **kwargs)
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.v_mse_loss = torch.nn.MSELoss()
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.lam = lam


    def log_epoch(self):
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('GradNormPi', average_only=True)
        self.logger.log_tabular('GradNormV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)

    def act(self, obs):
        return self.ac.step(obs)

    def compute_loss_pi(self, data):
        obs = data["obs"]
        act = data["act"]
        adv = data["adv"]
        logp_old = data["logp"]

        dist, logp = self.ac.pi(obs, act)

        kl = (logp_old - logp).mean().item()
        entropy = dist.entropy().mean().item()

        self.logger.store(KL=kl, Entropy=entropy)
        return -(logp * adv).mean()

    def compute_loss_v(self, data):
        obs = data["obs"]
        ret = data["ret"]

        v = torch.squeeze(self.ac.v(obs))
        return self.v_mse_loss(v, ret)

    def update(self):
        data = self.buf.get()

        self.ac.pi.zero_grad()
        pi_loss = self.compute_loss_pi(data)
        pi_loss.backward()
        pi_grad_norm = 0
        for p in self.ac.pi.parameters():
            pi_grad_norm += torch.norm(p.grad)
        self.pi_optimizer.step()

        v_grad_norm = 0
        for _ in range(self.train_v_iters):
            self.ac.v.zero_grad()
            v_loss = self.compute_loss_v(data)
            v_loss.backward()
            for p in self.ac.v.parameters():
                v_grad_norm += torch.norm(p.grad)
            self.vf_optimizer.step()

        v_grad_norm /= float(self.train_v_iters)

        self.logger.store(LossPi=pi_loss, LossV=v_loss, GradNormPi=pi_grad_norm, GradNormV=v_grad_norm)


def vpg(
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=None,
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        logger_kwargs=None,
        save_freq=10,
):
    """
    Vanilla Policy Gradient

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    ac_kwargs = ac_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    saved_config = locals()

    algo = VPGAlgorithm(
        env_fn=env_fn,
        actor_critic=actor_critic,
        ac_kwargs=ac_kwargs,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        pi_lr=pi_lr,
        vf_lr=vf_lr,
        train_v_iters=train_v_iters,
        lam=lam,
        max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs,
        saved_config=saved_config,
        save_freq=save_freq,
    )

    algo.run()
