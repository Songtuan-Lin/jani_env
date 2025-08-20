from jani_env import JaniEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


model_file = "examples/mod_linetrack/mod_linetrack.jani"
property_file = "examples/mod_linetrack/property.jani"
# start_file = "examples/mod_linetrack/inverted_pendulum/start.jani"
# goal_file = "examples/mod_linetrack/inverted_pendulum/objective.jani"
# safe_file = "examples/mod_linetrack/inverted_pendulum/safe.jani"

envs = make_vec_env(JaniEnv, n_envs=1, env_kwargs=dict(model_file=model_file, property_file=property_file))

model = PPO("MlpPolicy", envs, n_steps=100, verbose=1).learn(total_timesteps=1000000)