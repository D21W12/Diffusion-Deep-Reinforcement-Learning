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
        if len(x.shape) == 3:  # Transform single image
            diff = (x[1:] - x[:-1])
            dim = 0
        elif len(x.shape) == 4:  # Transform batch
            diff = (x[:, 1:] - x[:, :-1])
            dim = 1
        else:
            raise ValueError("Expects shape (C, H, W)")
        return torch.concat([x, diff.abs()], dim=dim)


class FrameChannelsOnly:
    def __init__(self, n: int = 4):
        self._n = n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:  # Transform single image
            return x[:self._n]
        elif len(x.shape) == 4:  # Transform batch
            return x[:, :self._n]
        else:
            raise ValueError("Expects shape (C, H, W)")


if __name__ == "__main__":
    transform = Difference()
    x = torch.randn((4, 88, 88))
    print(x.shape)
    x = transform(x)
    print(x.shape)
