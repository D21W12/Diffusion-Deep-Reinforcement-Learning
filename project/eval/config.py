from dataclasses import dataclass, field


@dataclass
class DiffEvalConfig:
    network: str = "edm2"
    resolution: int = 88
    in_channels: int = 4
    start_channels: int = 128
    num_res_blocks: int = 2
    channel_multipliers: list[int] | tuple[int] = field(default_factory=lambda: [2, 2, 2])
    attention_resolutions: list[int] | tuple[int] | set[int] = field(default_factory=lambda: [11])
    dropout: int = 0.13


@dataclass
class DQNEvalConfig:
    environment: str = "ALE/Breakout-v5"
