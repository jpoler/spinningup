from functools import partial
import time

from torch.optim import Adam
import numpy as np
import torch
import gym

from spinup.algos.mytorch.base.actor_critic import MLPActorCritic
from spinup.algos.mytorch.base.algorithm import Algorithm
from spinup.algos.mytorch.base.atari import is_atari_env
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
            train_iters=10,
            train_minibatches=4,
            lam=0.97,
            use_gpu=False,
            **kwargs,
    ):
        buf_fn = partial(GAEBuffer, gamma=gamma, lam=lam)

        super().__init__(env_fn, buf_fn, **kwargs)

        self.use_gpu = use_gpu
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        conv = is_atari_env(self.env)
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, conv=conv, **ac_kwargs)
        self.ac.to(self.device)
        self.v_mse_loss = torch.nn.MSELoss()
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_iters = train_iters
        self.train_minibatches = train_minibatches
        self.lam = lam


    def log_epoch(self):
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('GradNormPi', average_only=True)
        self.logger.log_tabular('GradNormV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('Advantage', average_only=True)

    def act(self, obs):
        return self.ac.step(obs, device=self.device)

    def compute_loss_pi(self, obs, act, adv, logp_old):
        dist, logp = self.ac.pi(obs, act)

        kl = (logp_old - logp).mean().item()
        entropy = dist.entropy().mean().item()

        self.logger.store(KL=kl, Entropy=entropy, Advantage=adv.mean().item())
        return -(logp * adv).mean()

    def compute_loss_v(self, obs, ret):
        v = self.ac.v(obs)
        return self.v_mse_loss(v, ret)

    def update(self):
        data = self.buf.get(device=self.device)
        obs_all = data["obs"]
        act_all = data["act"]
        ret_all = data["ret"]
        adv_all = data["adv"]
        logp_old_all = data["logp"]

        n_steps = len(obs_all)
        batch_size = n_steps // self.train_minibatches
        assert n_steps % batch_size == 0

        pi_grad_norms = []
        pi_losses = []
        v_grad_norms = []
        v_losses = []

        permutation = torch.randperm(n_steps)
        for j in range(0, n_steps, batch_size):
            idx = permutation[j:j+batch_size]
            obs, act, ret, adv, logp_old = obs_all[idx], act_all[idx], ret_all[idx], adv_all[idx], logp_old_all[idx]

            self.ac.pi.zero_grad()
            pi_loss = self.compute_loss_pi(obs, act, adv, logp_old)
            pi_loss.backward()
            pi_grad_norm = 0
            for p in self.ac.pi.parameters():
                if p.grad is not None:
                    pi_grad_norms.append(torch.norm(p.grad))
            self.pi_optimizer.step()
            pi_losses.append(pi_loss)

        for _ in range(self.train_iters):
            permutation = torch.randperm(n_steps)
            for j in range(0, n_steps, batch_size):
                idx = permutation[j:j+batch_size]
                obs, act, ret, adv, logp_old = obs_all[idx], act_all[idx], ret_all[idx], adv_all[idx], logp_old_all[idx]

                self.ac.v.zero_grad()
                v_loss = self.compute_loss_v(obs, ret)
                v_loss.backward()
                for p in self.ac.v.parameters():
                    v_grad_norms.append(torch.norm(p.grad))
                self.vf_optimizer.step()
                v_losses.append(v_loss)

        if len(pi_losses) > 0:
            pi_loss_total = sum(pi_losses) / float(len(pi_losses))
            pi_grad_norm = sum(pi_grad_norms) / float(len(pi_grad_norms))
            v_loss_total = sum(v_losses) / float(len(v_losses))
            v_grad_norm = sum(v_grad_norms) / float(len(v_grad_norms))
        else:
            pi_loss_total = 0
            pi_grad_norm = 0
            v_loss_total = 0
            v_grad_norm = 0

        self.logger.store(LossPi=pi_loss_total, LossV=v_loss_total, GradNormPi=pi_grad_norm, GradNormV=v_grad_norm)


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
        train_iters=10,
        train_minibatches=4,
        lam=0.97,
        max_ep_len=1000,
        logger_kwargs=None,
        save_freq=10,
        use_gpu=True,
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

        train_iters (int): Number of gradient descent steps to take
            per epoch.

        train_minibatches (int): Number of minibatches during training.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        use_gpu (bool): Which device to use for both acting and updating.

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
        train_iters=train_iters,
        train_minibatches=train_minibatches,
        lam=lam,
        max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs,
        saved_config=saved_config,
        save_freq=save_freq,
        use_gpu=use_gpu,
    )

    algo.run()
