from dataclasses import dataclass, field


@dataclass
class Config:
    checkpoint_path: str
    device: str
    lr: int
    batch_size: int


@dataclass
class DQNConfig(Config):
    memory_checkpoint_path: str
    frames: int
    model: str = str
    batch_size: int = 32
    lr: float = 2.5e-4
    discount: int = 0.99
    replay_size: int = 500_000
    environment: str = "ALE/Breakout-v5"
    target_update_frequency: int = 10_000
    final_exploration_frame: int = 1_000_000
    replay_start_size: int = 50_000


@dataclass
class DiffusionConfig(Config):
    data_path: str
    epochs: int
    network: str
    batch_size: int = 64
    lr: float = 2e-3
    resolution: int = 32
    in_channels: int = 3
    start_channels: int = 64
    num_res_blocks: int = 2
    channel_multipliers: list[int] | tuple[int] = field(default_factory=lambda: [1, 2, 2])
    attention_resolutions: list[int] | tuple[int] | set[int] = field(default_factory=lambda: [16])
    dropout: int = 0.13
