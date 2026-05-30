from .denoising_agent import DenoisingAgent
from ..util.filters import Filter


class NonLinearAgent(DenoisingAgent):

    def __init__(
            self,
            filter_: Filter
    ) -> None:
        self._denoiser = filter_
