import torch

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
    environment: str = "ALE/Breakout-v5"


@dataclass
class ExperimentConfig:
    seed: int = 2112
    environment: str = "ALE/Breakout-v5"
    episodes: int = 30
    N: int = 8
    epsilon: float = 0.05
    kernel_size: int = 3
    noise_levels: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
