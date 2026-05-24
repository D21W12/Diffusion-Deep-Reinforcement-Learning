from project.agents.denoising_agent import DenoisingAgent
from project.models import EDMMauMau, EDMEvelynn


class DiffusionAgent(DenoisingAgent):

    def __init__(
            self,
            model,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._denoiser = model

    def load(self, dqn, diffusion) -> 'DiffusionAgent':
        super().load(dqn)
        self._denoiser.load(diffusion)
        return self

    def to(
            self,
            device: str,
    ) -> 'DiffusionAgent':
        super().to(device)
        self._denoiser.to(device)
        return self