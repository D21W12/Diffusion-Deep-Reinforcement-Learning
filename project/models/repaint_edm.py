import torch

from project.models import EDMEvelynn


class EDMCallum(EDMEvelynn):

    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        pass