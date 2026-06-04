import torch
from torchvision import transforms

from project.agents import DQNAgent
from project.util.transforms import Difference, ToFloat, FrameChannelsOnly


class DiffusionAgent(DQNAgent):
    TRANSFORM = transforms.Compose([
        transforms.Pad(2),
        Difference(),
        transforms.Normalize(0.5, 0.5),
    ])
    INV_TRANSFORM = transforms.Compose([
        FrameChannelsOnly(4),
        transforms.CenterCrop(84),
        transforms.Normalize(-1, 2)
    ])

    def __init__(
            self,
            model,
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
        self._model = model

    def q_values(self, s) -> torch.Tensor:
        s = self.TRANSFORM(s)
        s = self._model.denoise(s.unsqueeze(0))
        s = self.INV_TRANSFORM(s)  # Apply the inverse of the transforms applied for the diffusion model
        return super().q_values(s.squeeze())

    def to(
            self,
            device: str,
    ) -> 'DiffusionAgent':
        super().to(device)
        self._model.to(device)
        return self