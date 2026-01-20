import tensordict

from torchrl.data import TensorDict, TensorDictReplayBuffer, LazyTensorStorage


class DAggerBuffer:
    """Replay buffer for DAgger algorithm."""

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        # Initialize storages for replay buffer
        self.positive_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
        ) # Buffer for safe state-action pairs
        self.negative_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=buffer_size),
        ) # Buffer for state-action pairs corrected by the oracle

    def add_samples(self, positive_samples: TensorDict, negative_samples: TensorDict):
        """Add samples to the respective buffers."""
        self.positive_buffer.extend(positive_samples)
        self.negative_buffer.extend(negative_samples)

    def sample(self, batch_size: int):
        """Sample a batch of state-action pairs from both buffers."""
        batch_size_neg = min(batch_size // 2, len(self.negative_buffer))
        batch_size_pos = batch_size - batch_size_neg
        pos_batch = self.positive_buffer.sample(batch_size_pos)
        neg_batch = self.negative_buffer.sample(batch_size_neg)
        # Ususally shuffle is not necessary since we are doing behavior cloning
        batch_data = tensordict.cat([pos_batch, neg_batch], dim=0)
        return batch_data