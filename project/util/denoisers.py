import torch
from scipy import ndimage


class NonLinearFilter:
    pass


class MedianFilter(NonLinearFilter):

    def __init__(
            self,
            kernel_size: int = 3
    ):
        self._kernel_size = kernel_size

    def denoise(self,  x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        y = ndimage.median_filter(x, self._kernel_size, output=x)
        return torch.from_numpy(y)


class GaussianFilter(NonLinearFilter):

    def __init__(
            self,
            sigma: float
    ):
        self._sigma = sigma

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        y = ndimage.gaussian_filter(x, self._sigma)
        return torch.from_numpy(y)