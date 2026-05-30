from project.agents.denoising_agent import DenoisingAgent


class DiffusionAgent(DenoisingAgent):

    def __init__(
            self,
            model,
            *args,
            **kwargs
    ):
        super().__init__(model, *args, **kwargs)

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