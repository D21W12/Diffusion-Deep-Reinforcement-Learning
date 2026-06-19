import torch
from tqdm import trange

from project.models import EDMEvelynn


class EDMDenoise(EDMEvelynn):
    
    def __init__(self, sigma_noise: float, img_resolution: int, img_channels: int, *args, **kwargs):
        super().__init__(img_resolution, img_channels, *args, **kwargs)

        self._sigma_noise = sigma_noise

    def set_sigma_noise(self, sigma_noise):
        self._sigma_noise = sigma_noise

    def set_N(self, N):
        self._N = N


class EDMMauMau(EDMDenoise):
    """Implementation of RePaint with EDM."""

    def __init__(self, sigma_noise: float, img_resolution: int, img_channels: int, *args, **kwargs):
        super().__init__(sigma_noise, img_resolution, img_channels, *args, **kwargs)

        self._sigma_noise = sigma_noise

    def denoise(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        self._score_network.eval()

        with torch.no_grad():

            sigma =  torch.full((x.shape[0],), self._sigma_noise, device=self._device)
            return self._D(x, sigma)
    

class EDMSerie(EDMDenoise):

    def __init__(self, sigma_noise: float, img_resolution: int, img_channels: int, *args, **kwargs):
        super().__init__(sigma_noise, img_resolution, img_channels, sigma_max=sigma_noise, *args, **kwargs)

    def set_sigma_noise(self, sigma_noise):
        self._sigma_noise = sigma_noise
        self._sigma_max = sigma_noise

    def denoise(self, x: torch.Tensor) -> torch.Tensor:
        
        self._score_network.eval()

        with torch.no_grad():

            x_next = x

            for i in range(self._N):

                x_next = self._heun_step(x_next, i)

        return x_next