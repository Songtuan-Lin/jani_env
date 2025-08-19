from jani_env import JaniEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


model_file = "examples/bouncing_ball.jani"
start_file = "examples/start_compact.jani"
goal_file = "examples/objective.jani"
safe_file = "examples/safe.jani"

env = JaniEnv(model_file, start_file, goal_file, safe_file)
envs = make_vec_env(JaniEnv, n_envs=5, env_kwargs=dict(model_file=model_file, start_file=start_file, goal_file=goal_file, safe_file=safe_file))

model = PPO("MlpPolicy", envs, n_steps=100, verbose=1).learn(total_timesteps=1000000)