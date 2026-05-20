import torch


class ToFloat:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255


class FP16Precision:

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.float16)


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


class Difference:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        diff = (x - x[1:]).abs()
        return torch.con([x, diff])