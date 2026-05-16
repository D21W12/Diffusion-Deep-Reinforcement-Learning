import torch

from project.agents import DQNAgent
from project.models import EDMEvelynn


class DiffusionAgent(DQNAgent):

    def __init__(
            self,
            n_actions: int,
            train: bool,
            lr: float = 2.5e-4,
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
            replay_size,
            discount,
            update_frequency,
            target_update_frequency,
            batch_size,
            replay_start_size,
            final_exploration_frame,
            device
        )

        self._loaded = False
        self._diffusion_model = None

    def select_action(self, o) -> int:

        s = self._diffusion_model.inpaint(o)
        return super().select_action(s)

    def load(
            self,
            dqn: str,
            diffusion: str,
    ) -> None:
        super().load(dqn)

        checkpoint = torch.load(diffusion, map_location=self._device)
        kwargs = checkpoint["config"]

        self._diffusion_model = EDMEvelynn(**kwargs)
        self._diffusion_model.load(diffusion)
        return self