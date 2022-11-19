import atari_py
import numpy as np

def is_atari_env(env):
    return hasattr(env, 'ale') and isinstance(env.ale, atari_py.ale_python_interface.ALEInterface)

def rgb_to_grayscale(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
