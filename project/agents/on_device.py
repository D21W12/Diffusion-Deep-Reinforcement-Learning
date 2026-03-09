import torch


class OnDevice:

    def __init__(self):
        self._device = "cpu"

    def to(self, device) -> 'OnDevice':
        OPTIONS = ["cuda", "mps", "cpu"]

        # Check if a correct/available device is given.
        if device not in OPTIONS:
            raise ValueError(f"Device should be one of {", ".join(OPTIONS)} not {device}!")
        elif device == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"Cuda not available!")
        elif device == "mps" and not torch.mps.is_available():
            raise ValueError(f"MPS not available!")

        self._device = device

        return self