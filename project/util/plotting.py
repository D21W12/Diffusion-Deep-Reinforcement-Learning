import torch
import matplotlib.pyplot as plt


def plot_sample(x: torch.Tensor) -> None:
    x = x.to("cpu")
    plt.imshow(x.mean(dim=0), cmap='grey')
    plt.show()
