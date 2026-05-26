import torch


def PSNR(
        y: torch.Tensor,
        y_hat: torch.Tensor,
        max_: torch.Tensor = torch.ones(1)
) -> float:
    mse = MSE(y, y_hat)
    return (20 * torch.log10(max_) - 10 * torch.log10(mse)).item()

def MSE(
        y: torch.Tensor,
        y_hat: torch.Tensor
) -> torch.Tensor:
    return (y - y_hat).pow(2).mean()
