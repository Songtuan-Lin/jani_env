import pandas as pd

from tensordict import TensorDict


def read_trajectories(file_path: str):
    df = pd.read_csv(file_path, header=None)
    obs = df.iloc[:-1, :-5].values
    actions = df.iloc[:, [-5]].values
    rewards = df.iloc[:, [-4]].values
    terminated = df.iloc[:, [-3]].values
    truncated = df.iloc[:, [-2]].values
    safety = df.iloc[:, [-1]].values
    next_obs = df.iloc[1:, :-5].values