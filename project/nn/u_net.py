import torch
import torch.nn as nn

from edm.training.networks import SongUNet


# TODO: NoiseEmbedding gets projected by an MLP to the number of channels
class NoiseEmbedding(nn.Module):

    def __init__(self, embedding_dims: int):
        super().__init__()

        self._embedding_dims = embedding_dims

    def forward(self, sigma):
        i = torch.arange(dim // 2)

        x = 1 / (10000 ** (2 * i / dim))
        x = torch.outer(sigma, x)

        embeddings = torch.zeros(sigma.shape + (self._embedding_dims,))
        embeddings[:, 0::2] = torch.sin(x)
        embeddings[:, 1::2] = torch.cos(x)

        return embeddings
