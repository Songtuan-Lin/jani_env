import pytest

import torch

from torchrl.envs import EnvBase, SerialEnv, ParallelEnv
from torchrl.data import Bounded, Binary, Categorical, Composite
from tensordict import TensorDict, TensorDictBase


class TestTorchRLEnv (EnvBase):
    def __init__(self):
        super().__init__()
        self.obs_dim = 2
        self.n_actions = 2
        self.observation_spec = Composite({
            "observation": Bounded(
                low=torch.tensor([-1.0] * self.obs_dim),
                high=torch.tensor([1.0] * self.obs_dim),
                shape=(self.obs_dim,)
            ),
            "action_mask": Binary(
                n=self.n_actions,
                shape=(self.n_actions,),
                dtype=torch.bool
            )
        })
        self.action_spec = Categorical(n=self.n_actions)
        self.reward_spec = Bounded(low=-1.0, high=1.0, shape=(1,))
        self.done_spec = Binary(n=1, shape=(1,))
        self.current_obs = None

    def action_mask(self) -> torch.Tensor:
        return torch.tensor([0, 1], dtype=torch.bool)

    def _reset(self, td: TensorDictBase) -> TensorDictBase:
        obs = torch.tensor([0.0] * self.obs_dim)
        self.current_obs = obs
        return TensorDict({"observation": obs, "action_mask": self.action_mask()}, batch_size=[])

    def _step(self, td: TensorDictBase) -> TensorDictBase:
        obs = self.current_obs + torch.tensor([0.0, 0.1])
        self.current_obs = obs
        reward = torch.tensor([0.01])
        # done = torch.tensor([torch.rand(1).item() > 0.95], dtype=torch.bool)  # 5% chance of being done
        done = torch.tensor([False], dtype=torch.bool)
        return TensorDict({
            "observation": obs,
            "reward": reward,
            "action_mask": self.action_mask(),
            "done": done
        }, batch_size=[])
    
    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng
    

class TestTorchRLEnvSuite:
    @pytest.fixture
    def env(self):
        return TestTorchRLEnv()

    def test_reset(self, env):
        td = env.reset()
        assert "observation" in td
        assert "action_mask" in td
        assert td["observation"].shape == (2,)

    def test_step(self, env):
        env.reset()
        action = torch.tensor([1])  # Valid action based on action mask
        td = TensorDict({"action": action}, batch_size=[])
        next_td = env.step(td)
        assert "observation" in next_td["next"]
        assert "reward" in next_td["next"]
        assert "done" in next_td["next"]
        assert next_td["next"]["observation"].shape == (2,)
        assert next_td["next"]["reward"].shape == (1,)
        assert next_td["next"]["done"].shape == (1,)
        assert (next_td["next"]["observation"] == torch.tensor([0.0, 0.1])).all()

    def test_rollout(self, env):
        td = env.rollout(max_steps=10)
        assert "observation" in td["next"]
        assert td["next"]["observation"].shape == (10, 2)  # 10 steps lead to 9 next observations
        expected_obs = torch.tensor([
            [0.0, 0.1],
            [0.0, 0.2],
            [0.0, 0.3],
            [0.0, 0.4],
            [0.0, 0.5],
            [0.0, 0.6],
            [0.0, 0.7],
            [0.0, 0.8],
            [0.0, 0.9],
            [0.0, 1.0]
        ])
        assert torch.allclose(td["next"]["observation"], expected_obs)


class TestTorchRLVectorEnvSuite:
    @pytest.fixture
    def serial_env(self):
        base_env = TestTorchRLEnv
        return SerialEnv(num_workers=8, create_env_fn=lambda: base_env())

    @pytest.fixture
    def parallel_env(self):
        base_env = TestTorchRLEnv
        return ParallelEnv(num_workers=8, create_env_fn=lambda: base_env())

    def test_serial_env_reset(self, serial_env):
        td = serial_env.reset()
        assert td.batch_size == torch.Size([8])
        assert "observation" in td
        assert "action_mask" in td

    def test_parallel_env_reset(self, parallel_env):
        td = parallel_env.reset()
        assert td.batch_size == torch.Size([8])
        assert "observation" in td
        assert "action_mask" in td

    def test_serial_env_step(self, serial_env):
        serial_env.reset()
        action = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])  # Valid actions based on action mask
        td = TensorDict({"action": action}, batch_size=[8])
        next_td = serial_env.step(td)
        assert next_td.batch_size == torch.Size([8])
        assert "observation" in next_td["next"]
        assert next_td["next"]["observation"].shape == (8, 2)
        assert "reward" in next_td["next"]
        assert "done" in next_td["next"]

    def test_parallel_env_step(self, parallel_env):
        parallel_env.reset()
        action = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])  # Valid actions based on action mask
        td = TensorDict({"action": action}, batch_size=[8])
        next_td = parallel_env.step(td)
        assert next_td.batch_size == torch.Size([8])
        assert "observation" in next_td["next"]
        assert "reward" in next_td["next"]
        assert "done" in next_td["next"]
    
    def test_serial_env_rollout(self, serial_env):
        td = serial_env.rollout(max_steps=5)
        assert td.batch_size == torch.Size([8, 5])
        assert "observation" in td["next"]
        assert td["next"]["observation"].shape == (8, 5, 2)

    def test_parallel_env_rollout(self, parallel_env):
        td = parallel_env.rollout(max_steps=5)
        assert td.batch_size == torch.Size([8, 5])
        assert "observation" in td["next"]
        assert td["next"]["observation"].shape == (8, 5, 2)