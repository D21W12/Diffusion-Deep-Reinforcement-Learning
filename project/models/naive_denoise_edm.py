import torch
from tqdm import trange

from project.models import EDMEvelynn


class EDMMauMau(EDMEvelynn):
    """Implementation of RePaint with EDM."""

    def __init__(self, sigma_noise: float, img_resolution: int, img_channels: int, *args, **kwargs):
        super().__init__(img_resolution, img_channels, *args, **kwargs)

        self._sigma_noise = sigma_noise
        self._starting_step = self._get_starting_step()

    def _get_starting_step(self) -> int:
        ts = torch.tensor([self._t(i) for i in range(0, self._N)])
        sigmas = self._sigma(ts)
        dist = (sigmas - torch.tensor(self._sigma_noise)).abs()
        return dist.argmin().item()

    def denoise(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        self._score_network.eval()

        with torch.no_grad():

            x_next = x

            for i in trange(self._starting_step, self._N):

                x_next = self._heun_step(x_next, i)

        return x_next

    def naive_denoise(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        self._score_network.eval()

        with torch.no_grad():

            sigma =  torch.full((x.shape[0],), self._sigma_noise, device=self._device)
            return self._D(x, sigma)