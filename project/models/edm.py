import os

import torch
import matplotlib.pyplot as plt

from torch.optim import Adam
from tqdm import tqdm, trange

from ..nn import UNetEvelynn, EDM2UNet
from ..util.evaluate import evaluate_to_image
from ..util.weights_and_biases import WandB


class EDMEvelynn:

    @classmethod
    def from_checkpoint(cls, f, device: str):
        checkpoint = torch.load(f, map_location=device)
        kwargs = checkpoint["config"]

        # Fix saving error I made in my code
        if isinstance(kwargs['channel_mult'][0], list):
            kwargs['channel_mult'] = kwargs['channel_mult'][0]

        model = cls(**kwargs)
        return model.load(f)

    def __init__(
            self,
            img_resolution: int,
            img_channels: int,
            model_channels: int = 192,
            channel_mult: tuple[int] | list[int] = [1,2,3,4],
            num_blocks: int = 3,
            attn_resolutions: set[int] | tuple[int] | list[int] = [16,8],
            dropout: float = 0.13,
            batch_size: int | None = None,
            lr: float = 1e-3,
            N: int = 32,
            sigma_min: float = 0.002,
            sigma_max: float = 80,
            sigma_data: float = 0.5,
            rho: int = 7,
            P_mean: float = -1.2,
            P_std: float = 1.2,
            network: str = "edm2",
            mixed_precision: bool = False,
    ) -> None:

        self._device = "cpu"

        self._img_resolution = img_resolution
        self._img_channels = img_channels
        self._model_channels = model_channels
        self._channel_mult = channel_mult
        self._attn_resolutions = attn_resolutions
        self._num_blocks = num_blocks
        self._mixed_precision = mixed_precision

        self._dtype = torch.float16 if self._mixed_precision else torch.float32

        self._batch_size = batch_size
        self._epochs = 0

        self._N = N
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sigma_data = sigma_data
        self._rho = rho
        self._P_mean = P_mean
        self._P_std = P_std

        # Initializing score network
        if network == "edm2":
            self._score_network = EDM2UNet(
                img_resolution=img_resolution,
                img_channels=img_channels,
                model_channels=model_channels,
                num_blocks=num_blocks,
                channel_mult=channel_mult,
                attn_resolutions=attn_resolutions,
            )
        elif network == "ddpm":
            self._score_network = UNetEvelynn(
                img_resolution=img_resolution,
                img_channels=img_channels,
                start_channels=model_channels,
                out_channels=img_channels,
                num_blocks=num_blocks,
                channel_mult=channel_mult,
                attn_resolutions=attn_resolutions,
                dropout=dropout
            )

        self._optimizer = Adam(
            params=self._score_network.parameters(),
            lr=lr,
        )

    def _t(self, i: int) -> float:
        if i < self._N:
            rho_inv = 1 / self._rho
            a = self._sigma_max**rho_inv
            b = (i / (self._N - 1)) * (self._sigma_min**rho_inv - self._sigma_max**rho_inv)
            return (a + b)**self._rho
        return 0

    def _s(self, t) -> float:
        return 1

    def _s_dot(self, t) -> float:
        return 0

    def _sigma(self, t) -> float:
        return t

    def _sigma_dot(self, t: torch.Tensor) -> float:
        return 1

    def _c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self._sigma_data**2 / (sigma**2 + self._sigma_data**2)

    def _c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self._sigma_data / (self._sigma_data**2 + sigma**2)**0.5

    def _c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1 / (sigma**2 + self._sigma_data**2)**0.5

    def _c_noise(self, sigmas: torch.Tensor) -> torch.Tensor:
        return (1 / 4) * torch.log(sigmas)

    def _sample_sigma(self, n: int) -> torch.Tensor:
        rnd_normal = torch.randn((n,), device=self._device)
        sigma = (rnd_normal * self._P_std + self._P_mean).exp()
        return sigma

    def _lambda(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self._sigma_data**2) / (sigma * self._sigma_data)**2

    def _dx_dt(
            self,
            x: torch.Tensor,
            t: float,
    ) -> torch.Tensor:

        batch_sigma = torch.full((x.shape[0],), self._sigma(t), device=self._device)

        a = (self._sigma_dot(t) / self._sigma(t) + self._s_dot(t) / self._s(t)) * x
        b = self._sigma_dot(t) * self._s(t) / self._sigma(t) * self._D(x / self._s(t), batch_sigma)

        return a - b

    def _F(
            self,
            x: torch.Tensor,
            sigma: torch.Tensor
    ) -> torch.Tensor:
        y = self._score_network(x, sigma)
        return y

    def _D(
            self,
            x: torch.Tensor,
            sigmas: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            c_skip = self._c_skip(sigmas)[:, None, None, None]
            c_out = self._c_out(sigmas)[:, None, None, None]
            c_in = self._c_in(sigmas)[:, None, None, None]
            c_noise = self._c_noise(sigmas)

        x_in = (c_in * x).to(self._dtype)
        return c_skip * x + c_out * self._F(x_in, c_noise).to(torch.float32)

    @staticmethod
    def _loss(D_yn, y, weights) -> torch.Tensor:
        return torch.mean(weights[:, None, None, None] * (D_yn - y) ** 2)

    def _run_batch(self, y: torch.Tensor):

        if self._batch_size is None:
            self._batch_size = y.shape[0]

        sigma = self._sample_sigma(n=y.shape[0])
        weights = self._lambda(sigma)

        n = torch.randn_like(y) * sigma[:, None, None, None]
        yn = y + n

        D_yn = self._D(yn, sigma)

        # Doing a gradient descent step
        self._optimizer.zero_grad()
        loss = self._loss(D_yn, y, weights)
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def _run_epoch(
            self,
            dataloader,
            print_loss: bool = False,
            wandb: WandB | None = None,
    ):

        loss = 0

        desc = f"Epoch {self._epochs + 1}"
        for batch in tqdm(dataloader, desc=desc):
            loss += self._run_batch(batch.to(self._device))

        self._epochs += 1

        loss /= len(dataloader.dataset)
        if print_loss: print(f"Loss: {loss:.5f}")
        if wandb: wandb.log(loss=loss)

    def train(
            self,
            epochs: int,
            dataloader,
            print_loss: bool = True,
            wandb: WandB | None = None,
            evaluation_dir: str | None = None
    ):
        self._score_network.train()
        for epoch in range(epochs):
            self._run_epoch(dataloader, print_loss=print_loss, wandb=wandb)
            if evaluation_dir:
                path = os.path.join(evaluation_dir, f"{self._epochs}.png")
                evaluate_to_image(self, path)

    def sample(self, batch_size: int = 1) -> torch.Tensor:

        self._score_network.eval()

        with torch.no_grad():

            x_i_shape = (batch_size, self._img_channels, self._img_resolution, self._img_resolution)
            x_i = torch.randn(x_i_shape, device=self._device)
            x_i = x_i * self._sigma(self._t(0)) * self._s(self._t(0))

            x_next = x_i

            for i in trange(0, self._N):

                # Take Euler step from t_i to t_(i + 1)
                d_i = self._dx_dt(x_next, self._t(i))
                x_prime = x_next + (self._t(i + 1) - self._t(i)) * d_i

                # Apply second order correction unless sigma goes to zero
                if self._sigma(self._t(i + 1)) != 0:
                    d_i_prime = self._dx_dt(x_prime, self._t(i + 1))
                    x_next =  x_next + (self._t(i + 1) - self._t(i)) * (0.5 * d_i + 0.5 * d_i_prime)
                else:
                    x_next = x_prime

            return x_next

    def to(self, device: str) -> 'EDM':
        self._score_network.to(device)
        self._device = device
        return self

    def config_dict(self) -> dict:
        return {
            "img_resolution": self._img_resolution,
            "img_channels": self._img_channels,
            "model_channels": self._model_channels,
            "channel_mult": self._channel_mult,
            "attn_resolutions": self._attn_resolutions,
            "num_blocks": self._num_blocks,
            "mixed_precision": self._mixed_precision
        }

    def save(self, f) -> None:
        torch.save({
            'score_network_state_dict': self._score_network.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'epochs': self._epochs,
            'config': self.config_dict()
        }, f)

    def load(self, f) -> 'EDMEvelynn':

        data = torch.load(f, map_location=self._device)

        self._score_network.load_state_dict(data["score_network_state_dict"])
        self._optimizer.load_state_dict(data["optimizer_state_dict"])
        self._epochs = data["epochs"]

        return self

