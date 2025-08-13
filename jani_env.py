import numpy as np
import gymnasium as gym

from jani import *


class JaniEnv(gym.Env):
    def __init__(self, model_file, start_file):
        super().__init__()
        self._jani = JANI(model_file, start_file)