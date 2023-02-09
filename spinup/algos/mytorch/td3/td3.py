from copy import deepcopy
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import numpy as np
import torch
import gym

from spinup.algos.mytorch.base.actor_critic import kaiming_uniform, MLPActorCritic
from spinup.algos.mytorch.base.algorithm import Algorithm
from spinup.algos.mytorch.base.buffer import ReplayBuffer
from spinup.utils.logx import EpochLogger

class TD3Algorithm(Algorithm):
    def __init__(
            self,
            env_fn,
            actor_critic=MLPActorCritic,
            ac_kwargs=None,
            epochs=50,
            replay_size=1e6,
            gamma=0.99,
            polyak=0.995,
            pi_lr=1e-3,
            q_lr=1e-3,
            batch_size=1e2,
            start_steps=1e4,
            update_after=1e3,
            update_every=50,
            act_noise=0.1,
            target_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            num_test_episodes=10,
            use_gpu=False,
            **kwargs,
    ):
        super().__init__(
            env_fn,
            ReplayBuffer,
            buf_size=replay_size,
            epochs=epochs,
            update_every=update_every,
            num_test_episodes=num_test_episodes,
            **kwargs,
        )

        self.use_gpu = use_gpu
        self.device = torch.device("cuda:0" if self.use_gpu else "cpu")
        self.ac = actor_critic(
            self.env.observation_space,
            self.env.action_space,
            activation=torch.nn.ReLU,
            actor_gaussian_noise=act_noise,
            deterministic=True,
            q_net=True,
            double_q=True,
            init=kaiming_uniform,
            **ac_kwargs,
        )
        print(self.ac)
        self.ac_target = deepcopy(self.ac)
        self.ac.to(self.device)
        self.ac_target.to(self.device)
        self.q_mse_loss = torch.nn.MSELoss()
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q1_optimizer = Adam(self.ac.v.parameters(), lr=q_lr)
        self.q2_optimizer = Adam(self.ac.v2.parameters(), lr=q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.num_test_episodes = num_test_episodes
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.action_low = torch.as_tensor(self.env.action_space.low, device=self.device)
        self.action_high = torch.as_tensor(self.env.action_space.high, device=self.device)


        for p in self.ac_target.parameters():
            p.requires_grad = False


    def log_epoch(self):
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('GradNormPi', average_only=True)
        self.logger.log_tabular('GradNormV', average_only=True)

    def act(self, obs):
        if len(self.buf) < self.start_steps:
            sample = self.env.action_space.sample()
            return sample, 0., 1.
        return self.ac.step(obs, device=self.device)

    def compute_loss_q(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            next_act = self.ac_target.pi(next_obs, deterministic=True)
            noise_mean = torch.zeros_like(next_act, device=self.device)
            noise = torch.normal(noise_mean, std=self.target_noise).clip(min=-self.noise_clip, max=self.noise_clip).to(self.device)
            next_act_noisy = (next_act + noise).clip(min=self.action_low, max=self.action_high)
            target_pred_1 = self.ac_target.v(next_obs, next_act_noisy)
            target_pred_2 = self.ac_target.v2(next_obs, next_act_noisy)
            target_pred = torch.minimum(target_pred_1, target_pred_2)

        target = rew + self.gamma * (1 - done) * target_pred
        pred_1 = self.ac.v(obs, act)
        pred_2 = self.ac.v2(obs, act)


        return self.q_mse_loss(pred_1, target) + self.q_mse_loss(pred_2, target)

    def compute_loss_pi(self, obs):
        act = self.ac.pi(obs, deterministic=True)
        q = self.ac.v(obs, act)
        return -q.mean()

    def update(self):
        if len(self.buf) < self.update_after:
            return

        pi_grad_norms = []
        pi_losses = []
        q_grad_norms = []
        q_losses = []

        # TODO revert from q to v to avoid confusion
        for i in range(self.update_every):
            data = self.buf.get(batch_size=self.batch_size, device=self.device)

            obs = data["obs"]
            act = data["act"]
            rew = data["rew"]
            next_obs = data["next_obs"]
            done = data["done"]

            self.q1_optimizer.zero_grad()
            self.q2_optimizer.zero_grad()

            q_loss = self.compute_loss_q(obs, act, rew, next_obs, done)
            q_loss.backward()
            for p in self.ac.v.parameters():
                q_grad_norms.append(torch.norm(p.grad))
            for p in self.ac.v2.parameters():
                q_grad_norms.append(torch.norm(p.grad))
            self.q1_optimizer.step()
            self.q2_optimizer.step()

            self.pi_optimizer.zero_grad()

            if i % self.policy_delay != 0:
                continue

            for p in self.ac.v.parameters():
                p.requires_grad = False

            pi_loss = self.compute_loss_pi(obs)
            pi_loss.backward()
            for p in self.ac.pi.parameters():
                if p.grad is not None:
                    pi_grad_norms.append(torch.norm(p.grad))

            self.pi_optimizer.step()

            for p in self.ac.v.parameters():
                p.requires_grad = True

            q_losses.append(q_loss)
            pi_losses.append(pi_loss)

            for (p_targ, p) in zip(self.ac_target.parameters(), self.ac.parameters()):
                # Ignore constant parameters
                if p.requires_grad == False:
                    continue
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


        pi_loss_total = sum(pi_losses) / float(len(pi_losses))
        pi_grad_norm = sum(pi_grad_norms) / float(len(pi_grad_norms))
        q_loss_total = sum(q_losses) / float(len(q_losses))
        q_grad_norm = sum(q_grad_norms) / float(len(q_grad_norms))

        self.logger.store(
            LossPi=pi_loss_total,
            LossV=q_loss_total,
            GradNormPi=pi_grad_norm,
            GradNormV=q_grad_norm,
        )


def td3(
        env_fn,
        actor_critic=MLPActorCritic,
        ac_kwargs=None,
        seed=0,
        steps_per_epoch=4000,
        epochs=50,
        replay_size=1e6,
        gamma=0.99,
        polyak=0.995,
        pi_lr=1e-3,
        q_lr=1e-3,
        batch_size=100,
        start_steps=1e4,
        update_after=1000,
        update_every=50,
        act_noise=0.1,
        target_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        num_test_episodes=10,
        max_ep_len=1000,
        logger_kwargs=None,
        save_freq=10,
        use_gpu=False,
):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        use_gpu (bool): Which device to use for both acting and updating.

    """
    ac_kwargs = ac_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    saved_config = locals()

    algo = TD3Algorithm(
        env_fn=env_fn,
        actor_critic=actor_critic,
        ac_kwargs=ac_kwargs,
        seed=seed,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        replay_size=replay_size,
        gamma=gamma,
        polyak=polyak,
        pi_lr=pi_lr,
        q_lr=q_lr,
        batch_size=batch_size,
        start_steps=start_steps,
        update_after=update_after,
        update_every=update_every,
        act_noise=act_noise,
        target_noise=target_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        num_test_episodes=num_test_episodes,
        max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs,
        saved_config=saved_config,
        save_freq=save_freq,
        use_gpu=use_gpu,
    )

    algo.run()
