import torch.nn as nn


class PrintShape(nn.Module):
    """
    Neural network module for printing the shape of the input data.

    Atr:
        _tag: tag to be printed with the shape (for easy identification).
    """

    def __init__(self, tag: str | None = None):
        super().__init__()

        self._tag = tag

    def forward(self, X):
        if self._tag:
            print(f"{self._tag}: {X.shape}")
            return X
        print(X.shape)
        return X
