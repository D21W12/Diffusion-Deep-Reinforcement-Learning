from .denoising_agent import DenoisingAgent
from ..util.denoisers import NonLinearFilter


class NonLinearAgent(DenoisingAgent):

    def __init__(
            self,
            filter_: NonLinearFilter
    ) -> None:
        self._denoiser = filter_
