import torch
import torch.nn as nn


class DQNMilkyWay(nn.Module):

    def __init__(self, n_actions: int):
        super().__init__()

        self._network = nn.Sequential(
            nn.LazyConv2d(
                out_channels=16,
                kernel_size=8,
                padding=0,
                stride=4
            ), nn.ReLU(),
            nn.LazyConv2d(
                out_channels=32,
                kernel_size=6,
                padding=0,
                stride=3
            ), nn.ReLU(),
            nn.LazyConv2d(
                out_channels=64,
                kernel_size=4,
                padding=0,
                stride=2
            ), nn.ReLU(), nn.Flatten(),
            nn.LazyLinear(
                out_features=512,
                bias=True
            ), nn.ReLU(),
            nn.LazyLinear(
                out_features=n_actions
            ),
        )

    def forward(self, X) -> torch.Tensor:
        return self._network(X)


if __name__ == "__main__":
    dqn = DQNMilkyWay(n_actions=4)
    s = torch.rand((1, 4, 210, 160))  # First dimension is the batch dimension
    dqn(s)
