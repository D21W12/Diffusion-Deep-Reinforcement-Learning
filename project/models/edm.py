import torch

from torch.optim import Adam
from tqdm import tqdm, trange

from ..nn import UNet


class EDM:

    def __init__(
            self,
            image_resolution: int,
            image_channels: int,
            batch_size: int | None = None,
            lr: float = 2e-4,
            N: int = 35,
            sigma_min: float = 0.002,
            sigma_max: float = 80,
            sigma_data: float = 0.5,
            rho: int = 7,
            P_mean: float = -1.2,
            P_std: float = 1.2,
    ) -> None:

        self._device = "cpu"

        self._image_resolution = image_resolution
        self._image_channels = image_channels

        self._batch_size = batch_size

        self._N = N
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sigma_data = sigma_data
        self._rho = rho
        self._P_mean = P_mean
        self._P_std = P_std

        self._score_network = UNet(
            img_resolution=image_resolution,
            img_channels=image_channels,
        )
        self._optimizer = Adam(
            params=self._score_network.parameters(),
            lr=lr
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

    def _c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return (1 / 4) * torch.log(torch.tensor(sigma, device=self._device))

    def _sample_sigma(self, n: int) -> torch.Tensor:
        rnd_normal = torch.randn((n,), device=self._device)
        sigma = (rnd_normal * self._P_std + self._P_mean).exp()
        return sigma

    def _lambda(self, sigma: torch.Tensor) -> float:
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
        return self._score_network(x, sigma)

    def _D(
            self,
            x: torch.Tensor,
            sigma: torch.Tensor
    ) -> torch.Tensor:

        x = self._c_in(sigma)[:, None, None, None] * x
        sigma = self._c_noise(sigma)

        skip = self._c_skip(sigma)[:, None, None, None] * x
        out = self._c_out(sigma)[:, None, None, None] * self._F(x, sigma)


        return skip + out

    @staticmethod
    def _loss(D_yn, y, weights) -> torch.Tensor:
        return torch.sum(weights[:, None, None, None] * ((D_yn - y) ** 2))

    def _training_step(self, y: torch.Tensor):

        if self._batch_size is None:
            self._batch_size = y.shape[0]

        sigma = self._sample_sigma(n=self._batch_size)
        weights = self._lambda(sigma)

        n = torch.randn_like(y) * sigma[:, None, None, None]
        yn = y + n

        D_yn = self._D(yn, sigma)

        # Doing a gradient descent step
        self._optimizer.zero_grad()
        loss = self._loss(D_yn, y, weights)
        loss.backward()
        self._optimizer.step()

    def _epoch(
            self,
            dataloader,
            epoch_ann: int | None = None
    ):
        desc = f"Epoch {epoch_ann}" if epoch_ann else "Epoch process"
        for batch, _ in tqdm(dataloader, desc=desc):
            self._training_step(batch.to(self._device))

    def train(
            self,
            epochs: int,
            dataloader,
    ):
        for epoch in range(epochs):
            self._epoch(dataloader)

    def heun_sampler(self, batch_size: int = 1) -> torch.Tensor:

        with torch.no_grad():

            # TODO: change the initial sample
            x_i = torch.normal(
                mean=0,
                std=self._sigma(self._t(0))**2 * self._s(self._t(0))**2,
                size=(batch_size,) + (self._image_channels, self._image_resolution, self._image_resolution),
                device=self._device
            )  # Generate initial sample

            for i in trange(0, self._N):

                # Take Euler step from t_i to t_(i + 1)
                d_i = self._dx_dt(x_i, self._t(i))
                x_prime = x_i + (self._t(i + 1) - self._t(i)) * d_i

                # Apply second order correction unless sigma goes to zero
                if self._sigma(self._t(i + 1)) != 0:
                    d_i_prime = self._dx_dt(x_prime, self._t(i + 1))
                    x_next =  x_i + (self._t(i + 1) - self._t(i)) * (0.5 * d_i + 0.5 * d_i_prime)
                else:
                    x_next = x_prime

            return x_next

    def to(self, device: str) -> 'EDM':
        self._score_network.to(device)
        self._device = device
        return self


if __name__ == "__main__":
    edm = EDM((1,))
    print(edm._t(0))
    print(edm._t(1))
    print(edm._t(2))
    print(edm._t(35))
