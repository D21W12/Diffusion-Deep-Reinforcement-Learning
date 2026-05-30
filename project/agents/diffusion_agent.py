from torchvision import transforms

from project.agents.denoising_agent import DenoisingAgent
from project.util.transforms import Difference, ToFloat


class DiffusionAgent(DenoisingAgent):
    TRANSFORM = transforms.Compose([
        ToFloat(),
        transforms.Pad(2),
        Difference(),
        transforms.Normalize(0.5, 0.5),
    ])

    def __init__(
            self,
            model,
            *args,
            **kwargs
    ):
        super().__init__(model, *args, **kwargs)

    def select_action(self, o) -> int:
        o = self.TRANSFORM(o)
        return super().select_action(o)

    def to(
            self,
            device: str,
    ) -> 'DiffusionAgent':
        super().to(device)
        self._denoiser.to(device)
        return self