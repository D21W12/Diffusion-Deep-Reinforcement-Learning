import math

import torch

from ..nn import SongUNet


class EDM:

    def __init__(
            self,
            obs_shape: tuple,
            N: int = 35,
            sigma_min: float = 0.002,
            sigma_max: float = 80,
            sigma_data: float = 0.5,
            rho: int = 7,
            P_mean: float = -1.2,
            P_std: float = 1.2,
    ) -> None:

        self._N = N
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._sigma_data = sigma_data
        self._rho = rho
        self._P_mean = P_mean
        self._P_std = P_std
        self._obs_shape = obs_shape

        self._score_network = SongUNet(
            in_channels=obs_shape[-1],
            out_channels=obs_shape[-1],
        )

    def _t(self, i: int) -> float:
        if i < self._N:
            rho_inv = 1 / self._rho
            a = self._sigma_max**rho_inv
            b = (i / (self._N - 1)) * (self._sigma_min**rho_inv - self._sigma_max**rho_inv)
            return (a + b)**self._rho
        return 0

    def _s(self, t: float) -> float:
        return 1

    def _s_dot(self, t: float) -> float:
        return 0

    def _sigma(self, t: float) -> float:
        return t

    def _sigma_dot(self, t: float) -> float:
        return 1

    def _c_skip(self, sigma: float) -> float:
        return self._sigma_data**2 / (sigma**2 + self._sigma_data**2)

    def _c_out(self, sigma: float) -> float:
        return sigma * self._sigma_data / (self._sigma_data**2 + sigma**2)**0.5

    def _c_in(self, sigma: float) -> float:
        return 1 / (sigma**2 + self._sigma_data**2)**0.5

    def _c_noise(self, sigma: float) -> float:
        return (1 / 4) * math.log(sigma)

    def sample_sigma(self) -> float:
        ln_sigma = torch.normal(mean=self._P_mean, std=self._P_std**2, size=(1,)).item()
        return math.exp(ln_sigma)

    def _lambda(self, sigma: float) -> float:
        return (sigma**2 + self._sigma_data**2) / (sigma * self._sigma_data)**2

    def _dx_dt(
            self,
            x: torch.Tensor,
            t: float,
    ) -> torch.Tensor:

        a = (self._sigma_dot(t) / self._sigma(t) + self._s_dot(t) / self._s(t)) * x
        b = self._sigma_dot(t) * self._s(t) / self._sigma(t) * self._D(x / self._s(t), self._sigma(t))

        return a - b

    def _F(
            self,
            x: torch.Tensor,
            sigma: float
    ) -> torch.Tensor:
        return self._score_network(x, sigma)

    def _D(
            self,
            x: torch.Tensor,
            sigma: float
    ) -> torch.Tensor:

        x = self._c_in(sigma) * x
        sigma = self._c_noise(sigma)

        skip = self._c_skip(sigma) * x
        out = self._c_out(sigma) * self._F(x, sigma)

        return skip + out

    def heun_sample(self, batch_size: int = 1) -> torch.Tensor:

        x = torch.zeros(size=(self._N,) + self._obs_shape)

        x[0] = torch.normal(
            mean=0,
            std=self._sigma(self._t(0))**2 * self._s(self._t(0))**2,
            size=(batch_size,) + self._obs_shape
        )  # Generate initial sample

        for i in range(0, self._N):

            # Take Euler step from t_i to t_(i + 1)
            d_i = self._dx_dt(x[:, i], self._t(i))
            x[:, i + 1] = x[:, i] + (self._t(i + 1) - self._t(i)) * d_i

            # Apply second order correction unless sigma goes to zero
            if self._sigma(self._t(i + 1)) != 0:
                d_i_prime = self._dx_dt(x[:, i + 1], self._t(i + 1))
                x[:, i + 1] =  x[:, i] + (self._t(i + 1) - self._t(i)) * (0.5 * d_i + 0.5 * d_i_prime)

        return x[:, self._N]


if __name__ == "__main__":
    edm = EDM((1,))
    print(edm._t(0))
    print(edm._t(1))
    print(edm._t(2))
    print(edm._t(35))
