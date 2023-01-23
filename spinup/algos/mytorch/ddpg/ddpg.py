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

def print_computation_graph(root, indent=0):
    if not root:
        return
    prefix = "  "*indent
    for fn, idx in root.next_functions:
        print(f"{prefix}fn: {fn}, idx: {idx}")
        if hasattr(fn, "variable"):
            print(f"{prefix}  var: {fn.variable.shape}")
        print_computation_graph(fn, indent=indent + 1)

class DDPGAlgorithm(Algorithm):
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
            q_weight_decay=0.,
            batch_size=1e2,
            start_steps=1e4,
            update_after=1e3,
            update_every=50,
            act_noise=0.1,
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
            init=kaiming_uniform,
            **ac_kwargs,
        )
        print(self.ac)
        self.ac_target = deepcopy(self.ac)
        self.ac.to(self.device)
        self.ac_target.to(self.device)
        self.q_mse_loss = torch.nn.MSELoss()
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.v.parameters(), lr=q_lr, weight_decay=q_weight_decay)
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
            target_pred = self.ac_target.v(next_obs, next_act)

        target = rew + self.gamma * (1 - done) * target_pred
        pred = self.ac.v(obs, act)

        return self.q_mse_loss(pred, target)

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

            self.q_optimizer.zero_grad()

            q_loss = self.compute_loss_q(obs, act, rew, next_obs, done)
            q_loss.backward()
            for p in self.ac.v.parameters():
                q_grad_norms.append(torch.norm(p.grad))
            self.q_optimizer.step()

            self.pi_optimizer.zero_grad()
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


def ddpg(
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
        q_weight_decay=0.,
        batch_size=100,
        start_steps=1e4,
        update_after=1000,
        update_every=50,
        act_noise=0.1,
        num_test_episodes=10,
        max_ep_len=1000,
        logger_kwargs=None,
        save_freq=10,
        use_gpu=False,
):
    ac_kwargs = ac_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    saved_config = locals()

    algo = DDPGAlgorithm(
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
        q_weight_decay=q_weight_decay,
        batch_size=batch_size,
        start_steps=start_steps,
        update_after=update_after,
        update_every=update_every,
        act_noise=act_noise,
        num_test_episodes=num_test_episodes,
        max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs,
        saved_config=saved_config,
        save_freq=save_freq,
        use_gpu=use_gpu,
    )

    algo.run()
