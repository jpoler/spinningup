## Spinning Up Re-implementation

My re-implementation of the six reinforcement learning algorithms featured in OpenAI's [Spinning Up](https://spinningup.openai.com/en/latest/index.html). 

Save for the implementation of the first algorithm, VPG, I generally tried to implement everything without looking at the reference code, using only the pseudocode on the Spinning Up site and the original whitepapers. Despite that I did borrow from the `ActorCritic` class during while implementing VPG.

## Code

[base](https://github.com/jpoler/spinningup/tree/master/spinup/algos/mytorch/base) contains packages that are shared by the actual algorithms. This includes the [abstract base class](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/base/algorithm.py) `Algorithm`, which implements the training loop itself. Each algorithm is responsible for implementing `update` and `act`. `update` contains the logic for updating model parameters according the specification each specific algorithm.

Algorithm implementations:

| [VPG](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/vpg/vpg.py) | [TRPO](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/trpo/trpo.py) | [PPO](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/ppo/ppo.py) | [DDPG](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/ddpg/ddpg.py) | [TD3](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/td3/td3.py) | [SAC](https://github.com/jpoler/spinningup/blob/master/spinup/algos/mytorch/sac/sac.py) |
| ---- | ---- | ---- | ---- | ---- | ---- |

## Results

Below are benchmarks for each of the 6 algorithms in 6 Mujoco environments. Each agent was allowed to learn with 3 random seeds, each seed exposed to 3 million total frames per environment. These benchmarks can be compared to the [Spinning Up Benchmarks](https://spinningup.openai.com/en/latest/spinningup/bench.html). Note that these benchmarks are not fair to the on-policy algorithms: they generally require significantly more experience to reach a comparable level of performance to off-policy algorithms, and so in most cases one could expect the on-policy algorithms to display better performance with say 10 million frames of experience. It would probably be more fair to allow all algorithms a chance to converge rather than artificially restricting total experience. This is justifiable because on-policy algorithms are generally less computationally intensive per update and faster in terms of wall-clock time. However, the Spinning Up benchmarks use 3 million frames of experience, so I followed suit.

Links to gifs of agent behavior are also included. Note that there is no attempt to cherry-pick agents with the best-performing random seed or particularly good episodes. These videos were recorded from the first random seed for each algorithm/environment for three episodes.

### Swimmer-v3

![swimmer benchmark](https://github.com/jpoler/spinningup/blob/master/graphs/swimmer.png)

| <img width=120> | VPG | TRPO | PPO | DDPG | TD3 | SAC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Swimmer-v3 | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/vpg/vpg_swimmer_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/trpo/trpo_swimmer_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ppo/ppo_swimmer_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ddpg/ddpg_swimmer_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/td3/td3_swimmer_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/sac/sac_swimmer_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> |

### Halfcheetah-v3

![halfcheetah benchmark](https://github.com/jpoler/spinningup/blob/master/graphs/halfcheetah.png)

| <img width=120> | VPG | TRPO | PPO | DDPG | TD3 | SAC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| HalfCheetah-v3 | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/vpg/vpg_halfcheetah_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/trpo/trpo_halfcheetah_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ppo/ppo_halfcheetah_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ddpg/ddpg_halfcheetah_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/td3/td3_halfcheetah_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/sac/sac_halfcheetah_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> |

### Hopper-v3

![hopper benchmark](https://github.com/jpoler/spinningup/blob/master/graphs/hopper.png)

| <img width=120> | VPG | TRPO | PPO | DDPG | TD3 | SAC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Hopper-v3 | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/vpg/vpg_hopper_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/trpo/trpo_hopper_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ppo/ppo_hopper_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ddpg/ddpg_hopper_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/td3/td3_hopper_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/sac/sac_hopper_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> |

### Walker2d-v3

![walker2d benchmark](https://github.com/jpoler/spinningup/blob/master/graphs/walker2d.png)

| <img width=120> | VPG | TRPO | PPO | DDPG | TD3 | SAC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Walker2d-v3 | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/vpg/vpg_walker2d_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/trpo/trpo_walker2d_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ppo/ppo_walker2d_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ddpg/ddpg_walker2d_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/td3/td3_walker2d_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/sac/sac_walker2d_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> |

### Ant-v3

![ant benchmark](https://github.com/jpoler/spinningup/blob/master/graphs/ant.png)

| <img width=120> | VPG | TRPO | PPO | DDPG | TD3 | SAC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Ant-v3 | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/vpg/vpg_ant_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/trpo/trpo_ant_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ppo/ppo_ant_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ddpg/ddpg_ant_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/td3/td3_ant_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/sac/sac_ant_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> |

### Humanoid-v1

![humanoid benchmark](https://github.com/jpoler/spinningup/blob/master/graphs/humanoid.png)

| <img width=120> | VPG | TRPO | PPO | DDPG | TD3 | SAC |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Humanoid-v1 | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/vpg/vpg_humanoid_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/trpo/trpo_humanoid_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ppo/ppo_humanoid_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/ddpg/ddpg_humanoid_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/td3/td3_humanoid_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> | <a href="https://github.com/jpoler/gifs/blob/main/spinningup/sac/sac_humanoid_new.gif"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Breezeicons-actions-22-media-playback-start.svg/1024px-Breezeicons-actions-22-media-playback-start.svg.png" height="32" width="32"></a> |

## Citation

The contents of this repository are based on [OpenAI's spinningup repository](https://github.com/openai/spinningup).

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```
