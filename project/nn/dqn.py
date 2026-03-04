import torch.nn as nn


class DQN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.LazyConv2d
        )
