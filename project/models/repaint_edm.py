import torch

from project.models import EDMEvelynn


class EDMMauMau(EDMEvelynn):

    def reconstruct(self, X: torch.Tensor) -> torch.Tensor:
        pass