import os

import numpy as np
import torch

from torch.utils.data import Dataset


class Sprites(Dataset):

    def __init__(
            self,
            images: str,
            labels: str,
            target: int,
            transform,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        labels = np.load(labels).argwhere()
        images = np.load(images)[labels == target]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)

        self._images = images
        self._transform = transform

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, item):
        image = self._images[item]
        if self._transform:
            image = self._transform(image)
        return image


if __name__ == "__main__":
    path = os.path.join("..", "..", "data", "sprites")
    images = np.load(os.path.join(path, "sprites.npy"))
    labels = np.load(os.path.join(path, "sprites_labels.npy"))
    print(images.permute(0, 3, 1, 2).shape)

