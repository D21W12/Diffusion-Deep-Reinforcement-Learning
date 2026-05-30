import torch

from project.agents import DQNAgent


class DenoisingAgent(DQNAgent):

    def __init__(
            self,
            denoiser,
            n_actions: int,
            train: bool,
            lr: float = 2.5e-4,
            epsilon: float = 0.05,
            replay_size: int = 1000000,
            discount: float = 0.99,
            update_frequency: int = 4,
            target_update_frequency: int = 10000,
            batch_size: int = 32,
            replay_start_size: int = 50000,
            final_exploration_frame: int = 1000000,
            device: str = "cpu"
    ) -> None:
        super().__init__(
            n_actions,
            train,
            lr,
            epsilon,
            replay_size,
            discount,
            update_frequency,
            target_update_frequency,
            batch_size,
            replay_start_size,
            final_exploration_frame,
            device
        )

        self._denoiser = denoiser

    def select_action(self, s) -> int:

        if not self._denoiser:
            raise Exception("No denoiser is initialized")

        s = self._denoiser.denoise(s)
        return super().select_action(s)
