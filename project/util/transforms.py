import torch


class ToFloat:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255


class PermuteChannels:
    """
    Permute channel dimensions from [H, W, C] to [C, H, W].
    """
    def __init__(self, permutation=None):
        if permutation is None:
            permutation = [2, 0, 1]
        self._permutation = permutation

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self._permutation)


class CastTo:
    def __init__(self, dtype) -> None:
        self._dtype = dtype

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self._dtype)