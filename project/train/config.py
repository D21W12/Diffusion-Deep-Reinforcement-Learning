from dataclasses import dataclass, field


class TrainingConfig:
    seed: int = 2112


@dataclass
class DQNTrainingTrainingConfig(TrainingConfig):
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
class DDQNTrainingTrainingConfig(TrainingConfig):
    device: str
    checkpoint_path: str
    memory_checkpoint_path: str
    lr: float = 2.5e-5
    batch_size: int = 32
    discount: int = 0.99
    replay_size: int = 1_000_000
    environment: str = "ALE/Seaquest-v5"
    target_update_frequency: int = 10_000
    final_exploration_frame: int = 1_000_000
    replay_start_size: int = 50_000


@dataclass
class DiffTrainingTrainingConfig(TrainingConfig):
    network: str = 'edm2'
    cap: int = 500_000
    batch_size: int = 32
    lr: float = 1e-3
    img_resolution: int = 88
    img_channels: int = 7
    model_channels: int = 64
    num_blocks: int = 3
    channel_mult: list[int] | tuple[int] = field(default_factory=lambda: [2, 2, 2])
    attn_resolutions: list[int] | tuple[int] | set[int] = field(default_factory=lambda: [22])
    dropout: int = 0.13
    mixed_precision: bool = False
