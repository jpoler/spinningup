from copy import deepcopy
from functools import partial
import time

from torch.optim import Adam
import torch
import gym

from spinup.algos.mytorch.base.actor_critic import MLPActorCritic
from spinup.algos.mytorch.base.algorithm import Algorithm
from spinup.algos.mytorch.base.buffer import GAEBuffer
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
import spinup.algos.mytorch.trpo.core as core

class TRPOAlgorithm(Algorithm):
    def __init__(
            self,
            env_fn,
            actor_critic=MLPActorCritic,
            ac_kwargs=None,
            gamma=0.99,
            delta=0.01,
            vf_lr=1e-3,
            train_v_iters=80,
            damping_coeff=0.1,
            cg_iters=10,
            backtrack_iters=10,
            backtrack_coeff=0.8,
            lam=0.97,
            **kwargs,
    ):
        buf_fn = partial(GAEBuffer, gamma=gamma, lam=lam)
        super().__init__(env_fn, buf_fn, **kwargs)
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)
        self.v_mse_loss = torch.nn.MSELoss()
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.gamma = gamma
        self.delta = delta
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.lam = lam


    def log_epoch(self):
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('GradNormPi', average_only=True)
        self.logger.log_tabular('GradNormV', average_only=True)
        self.logger.log_tabular('DNorm', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('AverageKL', average_only=True)
        self.logger.log_tabular('SurrogateAdvantage', average_only=True)
        self.logger.log_tabular('XHatCoefficient', average_only=True)
        self.logger.log_tabular('BacktrackIters', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)

    def act(self, obs):
        return self.ac.step(obs)

    def update(self):
        data = self.buf.get()
        obs = data["obs"]
        act = data["act"]
        ret = data["ret"]
        adv = data["adv"]

        actor_new = deepcopy(self.ac.pi)
        with torch.no_grad():
            _, logp_old = self.ac.pi(obs, act)
        pi_new, logp_new = self.ac.pi(obs, act)
        entropy = pi_new.entropy().mean().item()
        pi_loss = core.surrogate_advantage(logp_old, logp_new, adv)
        grads = torch.autograd.grad(pi_loss, self.ac.pi.parameters())
        pi_loss_g = torch.nn.utils.parameters_to_vector(grads)
        pi_grad_norm = torch.norm(pi_loss_g)
        x_hat, x_hat_coeff = core.compute_direction(
            obs, act, adv, pi_loss_g, self.ac.pi, actor_new, self.cg_iters, self.damping_coeff, self.delta)
        d = x_hat_coeff * x_hat
        d_norm = torch.norm(d)
        self.ac.pi, average_kl, surrogate_advantage, backtrack_iters_actual = core.line_search(
            obs, act, adv, d, self.backtrack_iters, self.backtrack_coeff, self.delta, self.ac.pi, actor_new)

        v_grad_norm = 0
        for _ in range(self.train_v_iters):
            self.ac.v.zero_grad()
            v = self.ac.v(obs)
            v_loss = self.v_mse_loss(v, ret)
            v_loss.backward()
            for p in self.ac.v.parameters():
                v_grad_norm += torch.norm(p.grad)
            self.vf_optimizer.step()

        v_grad_norm /= float(self.train_v_iters)

        self.logger.store(LossPi=pi_loss, LossV=v_loss, Entropy=entropy, GradNormV=v_grad_norm, GradNormPi=pi_grad_norm, DNorm=d_norm, AverageKL=average_kl, SurrogateAdvantage=surrogate_advantage, XHatCoefficient=x_hat_coeff, BacktrackIters=backtrack_iters_actual)



def trpo(
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=None,
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        delta=0.01,
        vf_lr=1e-3,
        train_v_iters=80,
        damping_coeff=0.1,
        cg_iters=10,
        backtrack_iters=10,
        backtrack_coeff=0.8,
        lam=0.97,
        max_ep_len=1000,
        logger_kwargs=None,
        save_freq=10,
):
    """
    Trust Region Policy Optimization

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

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update.
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be
            smallish. Adjusts Hessian-vector product calculation:

            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient.
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform.
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down.

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the
            backtracking line search. Since the line search usually doesn't
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

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

    algo = TRPOAlgorithm(
        env_fn=env_fn,
        actor_critic=actor_critic,
        ac_kwargs=ac_kwargs,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        delta=delta,
        vf_lr=vf_lr,
        train_v_iters=train_v_iters,
        damping_coeff=damping_coeff,
        cg_iters=cg_iters,
        backtrack_iters=backtrack_iters,
        backtrack_coeff=backtrack_coeff,
        lam=lam,
        max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs,
        saved_config=saved_config,
        save_freq=save_freq,
    )

    algo.run()
