from dataclasses import dataclass, field

@dataclass
class DQNTrainingConfig:
    device: str
    checkpoint_path: str
    memory_checkpoint_path: str
    lr: float = 1e-4
    batch_size: int = 32
    discount: int = 0.99
    replay_size: int = 1_000_000
    environment: str = "ALE/Breakout-v5"
    target_update_frequency: int = 10_000
    final_exploration_frame: int = 1_000_000
    replay_start_size: int = 50_000


@dataclass
class DiffTrainingConfig:
    network: str = 'edm2'
    batch_size: int = 32
    lr: float = 2e-3
    resolution: int = 84
    in_channels: int = 4
    start_channels: int = 64
    num_res_blocks: int = 2
    channel_multipliers: list[int] | tuple[int] = field(default_factory=lambda: [1, 2])
    attention_resolutions: list[int] | tuple[int] | set[int] = field(default_factory=lambda: [21])
    dropout: int = 0.13
