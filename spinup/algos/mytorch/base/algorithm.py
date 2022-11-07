
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.mytorch.trpo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class Algorithm(ABC):
    def __init__(
            self,
            env,
            buf,
            ac,
            seed=0,
            steps_per_epoch=4000,
            epochs=50,
            max_ep_len=1000,
            save_freq=10,
            logger_kwargs=None,
            saved_config=None,
            **kwargs,
    ):
        self._logger_kwargs = logger_kwargs or {}
        self.logger = EpochLogger(**logger_kwargs)

        self.env = env
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        self.buf = buf

        self.ac = ac

        self.seed = seed + 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        self.local_steps_per_epoch = int(steps_per_epoch / num_procs())

        self.saved_config = saved_config
        self.save_freq = save_freq


    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def log_epoch(self):
        pass

    def _log_epoch(self, epoch, start_time):
        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
        self.logger.log_tabular('Time', time.time()-start_time)

        self.log_epoch()
        self.logger.dump_tabular()

    def run(self):
        # torch.set_printoptions(profile="full")
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # save configuration
        self.logger.save_config(self.saved_config)


        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)
        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)


        # Prepare for interaction with environment
        start_time = time.time()
        obs, ep_ret, ep_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.logger.store(Epoch=epoch)
            for t in range(self.local_steps_per_epoch):
                if t % 10000 == 0:
                    print(f"\repoch: ({epoch + 1}/{self.epochs}) steps: ({t}/{self.local_steps_per_epoch})", end="")
                if t == self.local_steps_per_epoch - 1:
                    print()
                act, val, logp = self.ac.step(obs)
                # self.logger.store(VVals=val)
                obs, rew, done, _ = self.env.step(act)
                ep_ret += rew
                ep_len += 1
                self.buf.store(obs, act, rew, val, logp)

                truncated = (ep_len >= self.max_ep_len) or (t == self.local_steps_per_epoch - 1)
                finished = done or truncated

                if truncated:
                    _, val, _ = self.ac.step(obs)
                    self.buf.finish_path(last_val=val)
                elif done:
                    self.buf.finish_path()

                if finished:
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    obs = self.env.reset()
                    ep_len = 0
                    ep_ret = 0

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self.logger.save_state({'env': self.env}, None)

            epoch_data = self.buf.get()

            self.update(epoch_data)

            self._log_epoch(epoch, start_time)

