import torch
from demucs.demucs import Demucs
from madre import register
from madre.extra.torch.models.torch_base_model import TorchBaseModel

TARGETS = ["vocals", "drums", "bass", "other", "accompaniment"]


@register()
class DemucsV2(TorchBaseModel):
    def __init__(
        self,
        sources: list[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(sources, str):
            if sources == "all":
                self.sources = TARGETS[:4]
            else:
                self.sources = [sources]
        else:
            self.sources = sources

        self.model = Demucs(sources=self.sources)

    def inference(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
