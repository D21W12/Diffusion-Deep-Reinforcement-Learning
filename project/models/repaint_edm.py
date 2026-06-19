import torch
from tqdm import trange

from project.models import EDMEvelynn


class EDMCallum(EDMEvelynn):
    """Implementation of RePaint with EDM."""

    def __init__(self, U: int, img_resolution: int, img_channels: int, *args, **kwargs):
        super().__init__(img_resolution, img_channels, *args, **kwargs)

        self._U = U

    def set_U(self, U: int) -> None:
        self._U = U

    def inpaint(
            self,
            x: torch.Tensor,
            mask: torch.Tensor
    ) -> torch.Tensor:

        self._score_network.eval()

        with torch.no_grad():

            x_next = torch.randn_like(x, device=self._device)
            x_next *= self._sigma(self._t(0)) * self._s(self._t(0))

            for i in trange(0, self._N):

                for u in range(0, self._U):

                    if i < (self._N - 1):
                        x_known = self._forward_step(x, i)
                    else:
                        x_known = x

                    x_unknown = self._heun_step(x_next, i)

                    x_next = mask * x_known + (1 - mask) * x_unknown

                    if u < (self._U - 1) and i < (self._N - 1):
                        x_next = self._diffusion_step(x_next, i)

        return x_next

    def _forward_step(
            self,
            x_0: torch.Tensor,
            i: int,
    ) -> torch.Tensor:
        return x_0 + self._sigma(self._t(i)) * torch.randn_like(x_0)

    def _diffusion_step(
            self,
            x: torch.Tensor,
            i: int,
    ) -> torch.Tensor:

        t = self._t(i + 1)
        t_prev = self._t(i)
        sigma = torch.full((x.shape[0],), self._sigma(t), device=self._device)
        sigma_prev = torch.full((x.shape[0],), self._sigma(t_prev), device=self._device)

        return x + torch.sqrt(sigma_prev ** 2 - sigma ** 2)[:, None, None, None] * torch.randn_like(x)