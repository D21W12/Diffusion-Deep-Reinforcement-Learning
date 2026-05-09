import os

import numpy as np
import torch

from torch.utils.data import Dataset


class ReplayMemoryData(Dataset):

    def __init__(
            self,
            memory: str,
            transform,
            cap: int | None = None,
            device: str = "cpu",
            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        state_dict = torch.load(memory, map_location=device)

        full = state_dict["full"]
        images = state_dict["s"]

        if not full:
            i = state_dict["i"]
            images = images[:i]
        if cap:
            images = images[-cap:]

        self._images = images.permute(0, 2, 3, 1).numpy()
        self._transform = transform
        self._device = device

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, item):
        image = self._images[item]
        if self._transform:
            image = self._transform(image)
        return image

    def to(self, device: str) -> 'ReplayMemoryData':
        self._images = self._images.to(device)
        self._device = device
        return self


class Sprites(Dataset):

    def __init__(
            self,
            images: str,
            transform,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        images = np.load(images)

        self._images = images
        self._transform = transform

    def __len__(self):
        return self._images.shape[0]

    def __getitem__(self, item):
        image = self._images[item]
        if self._transform:
            image = self._transform(image)
        return image


class SpritesSubset(Dataset):

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

        mask = np.argwhere(np.load(labels))[:, 1] == target
        images = np.load(images)
        images = images[mask]

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
