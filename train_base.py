import numpy as np

from jani_env import JaniEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_mask()

def make_env(file_args):
    env = JaniEnv(**file_args)
    env = ActionMasker(env, mask_fn)
    return env

model_file = "examples/mod_linetrack/mod_linetrack.jani"
property_file = "examples/mod_linetrack/property.jani"
# start_file = "examples/mod_linetrack/start.jani"
# goal_file = "examples/mod_linetrack/objective.jani"
# safe_file = "examples/mod_linetrack/safe.jani"

# envs = make_vec_env(JaniEnv, n_envs=4, env_kwargs=dict(model_file=model_file, start_file=start_file, goal_file=goal_file, safe_file=safe_file), wrapper_class=ActionMasker, wrapper_kwargs=dict(action_mask_fn=mask_fn))
envs = make_vec_env(lambda: make_env(dict(model_file=model_file, property_file=property_file, random_init=False)), n_envs=4)


model = MaskablePPO(MaskableActorCriticPolicy, envs, n_steps=100, verbose=1).learn(total_timesteps=1000000)