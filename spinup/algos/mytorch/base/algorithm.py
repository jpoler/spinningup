from abc import ABC, abstractmethod
import base64
import cloudpickle
import time
import zlib

import gym
import numpy as np
import torch

from spinup.utils.logx import EpochLogger

class Algorithm(ABC):
    def __init__(
            self,
            env_fn,
            buf_fn,
            seed=0,
            steps_per_epoch=4000,
            epochs=50,
            max_ep_len=1000,
            save_freq=10,
            logger_kwargs=None,
            saved_config=None,
            status_freq=1000,
            **kwargs,
    ):
        self._logger_kwargs = logger_kwargs or {}
        self.logger = EpochLogger(**logger_kwargs)

        self.env = env_fn()
        pickled_env_fn = cloudpickle.dumps(env_fn)
        self.encoded_env_fn = base64.b64encode(zlib.compress(pickled_env_fn)).decode('utf-8')


        # TODO revert
        # self.seed = seed + 10000 * proc_id()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.max_ep_len = max_ep_len
        # self.local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.local_steps_per_epoch = steps_per_epoch

        self.buf = buf_fn(self.env.observation_space.shape, self.env.action_space.shape, self.local_steps_per_epoch)

        self.saved_config = saved_config
        self.save_freq = save_freq

        self.status_freq = status_freq


    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def log_epoch(self):
        pass

    @abstractmethod
    def act(self):
        pass

    def _log_epoch(self, epoch, start_time):
        # Log info about epoch
        self.logger.log_tabular('Epoch', epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.steps_per_epoch)
        self.logger.log_tabular('Vals', with_min_and_max=True)
        self.logger.log_tabular('Time', time.time()-start_time)

        self.log_epoch()
        self.logger.dump_tabular()

    def _eval_lazyframe(self, obs):
        if isinstance(obs, gym.wrappers.frame_stack.LazyFrames):
            return obs.__array__()[np.newaxis, :]
        return obs

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
        obs = self._eval_lazyframe(self.env.reset())
        ep_ret, ep_len = 0, 0



        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(self.epochs):
            self.logger.store(Epoch=epoch)
            for t in range(self.local_steps_per_epoch):
                if t % self.status_freq == 0:
                    print(f"\repoch: ({epoch + 1}/{self.epochs}) steps: ({t}/{self.local_steps_per_epoch})", end="")
                if t == self.local_steps_per_epoch - 1:
                    print(f"\repoch: ({epoch + 1}/{self.epochs}) steps: ({t+1}/{self.local_steps_per_epoch})")
                act, val, logp = self.act(obs)
                self.logger.store(Vals=val)
                next_obs, rew, done, _ = self.env.step(act)
                ep_ret += rew
                ep_len += 1
                self.buf.store(obs, act, rew, val, logp)

                obs = self._eval_lazyframe(next_obs)

                truncated = (ep_len >= self.max_ep_len) or (t == self.local_steps_per_epoch - 1)
                finished = done or truncated

                if truncated:
                    _, val, _ = self.act(obs)
                    self.buf.finish_path(last_val=val)
                elif done:
                    self.buf.finish_path()

                if finished:
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                    obs = self._eval_lazyframe(self.env.reset())
                    ep_len = 0
                    ep_ret = 0

            # Save model
            if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self.logger.save_state({'env_fn': self.encoded_env_fn}, None)

            self.update()

            self._log_epoch(epoch, start_time)

