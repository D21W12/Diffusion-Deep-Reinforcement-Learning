from dataclasses import dataclass, field


@dataclass
class DiffEvalConfig:
    network: str = "edm2"
    resolution: int = 84
    in_channels: int = 4
    start_channels: int = 64
    num_res_blocks: int = 2
    channel_multipliers: list[int] | tuple[int] = field(default_factory=lambda: [1, 2])
    attention_resolutions: list[int] | tuple[int] | set[int] = field(default_factory=lambda: [21])
    dropout: int = 0.13


@dataclass
class DQNEvalConfig:
    environment: str = "ALE/SpaceInvaders-v5"


@dataclass
class ExperimentConfig:
    environment: str = "ALE/Breakout-v5"
    episodes: int = 30
    sigma_noise: float = 0.5
    N: int = 8
    epsilon: float = 0.05
    kernel_size: int = 3
