import torch


def get_available_acc() -> str:
    device = "cpu"
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    return device