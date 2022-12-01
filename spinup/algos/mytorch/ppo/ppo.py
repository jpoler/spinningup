from copy import deepcopy
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

from spinup.algos.mytorch.base.atari import is_atari_env

class PPOAlgorithm(Algorithm):
    def __init__(
            self,
            env_fn,
            actor_critic=MLPActorCritic,
            ac_kwargs=None,
            gamma=0.99,
            clip_ratio=0.2,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_pi_iters=80,
            train_v_iters=80,
            lam=0.97,
            target_kl=0.01,
            entropy_bonus_coef=0.01,
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
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.target_kl = target_kl
        self.entropy_bonus_coef = entropy_bonus_coef


    def log_epoch(self):
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('GradNormPi', average_only=True)
        self.logger.log_tabular('GradNormV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('AverageKL', average_only=True)
        self.logger.log_tabular('ClipFrac', average_only=True)

    def act(self, obs):
        return self.ac.step(obs, device=self.device)

    def average_kl(self, pi_old, pi_new):
        kl = torch.distributions.kl.kl_divergence(pi_old, pi_new)
        return kl.mean()

    def compute_loss_pi(self, obs, act, adv, logp_old):
        pi_new, logp_new = self.ac.pi(obs, act)
        ratio = torch.exp(logp_new - logp_old)
        adv_clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        surrogate_objective_clipped = -(torch.min(ratio * adv, adv_clipped)).mean()

        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        kl = (logp_old - logp_new).mean().item()
        entropy = pi_new.entropy().mean().item()
        self.logger.store(KL=kl, Entropy=entropy, ClipFrac=clip_frac)

        return surrogate_objective_clipped + self.entropy_bonus_coef * entropy

    def compute_loss_v(self, obs, ret):
        v = self.ac.v(obs)
        return self.v_mse_loss(v, ret)

    def update(self):
        data = self.buf.get(device=self.device)
        obs = data["obs"]
        act = data["act"]
        ret = data["ret"]
        adv = data["adv"]
        logp_old = data["logp"]

        actor_old = deepcopy(self.ac.pi)
        with torch.no_grad():
            pi_old, _ = actor_old(obs)

        pi_grad_norms = []
        pi_losses = []
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            pi_loss = self.compute_loss_pi(obs, act, adv, logp_old)
            pi_losses.append(pi_loss)

            saved_params = [deepcopy(p) for p in self.ac.pi.parameters()]
            state_dict = self.pi_optimizer.state_dict()

            pi_loss.backward()
            for p in self.ac.pi.parameters():
                pi_grad_norms.append(torch.norm(p.grad))
            self.pi_optimizer.step()

            with torch.no_grad():
                pi_new, _ = self.ac.pi(obs)
            akl = self.average_kl(pi_old, pi_new)

            if akl > 1.5*self.target_kl:
                self.logger.log(f"Average KL {akl} is larger than {1.5*self.target_kl}")
                for p, saved_p in zip(self.ac.pi.parameters(), saved_params):
                    p.data = saved_p.data
                self.pi_optimizer.load_state_dict(state_dict)
                break

        pi_stop_iter = i

        pi_loss_total = sum(pi_losses) / float(len(pi_losses))
        pi_grad_norm = sum(pi_grad_norms) / float(len(pi_grad_norms))

        v_grad_norms = []
        v_losses = []
        for _ in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            v_loss = self.compute_loss_v(obs, ret)
            v_losses.append(v_loss)
            v_loss.backward()
            for p in self.ac.v.parameters():
                v_grad_norms.append(torch.norm(p.grad))
            self.vf_optimizer.step()

        v_loss_total = sum(v_losses) / float(len(v_losses))
        v_grad_norm = sum(v_grad_norms) / float(len(v_grad_norms))

        self.logger.store(LossPi=pi_loss_total, LossV=v_loss_total, GradNormPi=pi_grad_norm, GradNormV=v_grad_norm, StopIter=pi_stop_iter, AverageKL=akl)

def ppo(
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=None,
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=3e-4,
        vf_lr=1e-3,
        train_pi_iters=80,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=1000,
        target_kl=0.01,
        entropy_bonus_coef=0.01,
        logger_kwargs=None,
        save_freq=10,
        use_gpu=False,
):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

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
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    ac_kwargs = ac_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    saved_config = locals()

    algo = PPOAlgorithm(
        env_fn=env_fn,
        actor_critic=actor_critic,
        ac_kwargs=ac_kwargs,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        gamma=gamma,
        clip_ratio=clip_ratio,
        pi_lr=pi_lr,
        vf_lr=vf_lr,
        train_pi_iters=train_pi_iters,
        train_v_iters=train_v_iters,
        lam=lam,
        max_ep_len=max_ep_len,
        target_kl=target_kl,
        entropy_bonus_coef=entropy_bonus_coef,
        logger_kwargs=logger_kwargs,
        saved_config=saved_config,
        save_freq=save_freq,
        use_gpu=use_gpu,
    )

    algo.run()
