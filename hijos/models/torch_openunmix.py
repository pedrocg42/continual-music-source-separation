import numpy as np
import torch
from madre import register
from madre.extra.torch.models.torch_base_model import TorchBaseModel
from openunmix.model import OpenUnmix

TARGETS = ["vocals", "drums", "bass", "other", "accompaniment"]


@register()
class TorchOpenUnmixModel(TorchBaseModel):
    def __init__(
        self,
        nb_bins: int = 4096,
        nb_channels: int = 2,
        hidden_size: int = 512,
        nb_layers: int = 3,
        unidirectional: bool = False,
        input_mean: np.ndarray | None = None,
        input_scale: np.ndarray | None = None,
        max_bin: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.model = OpenUnmix(
            nb_bins=nb_bins,
            nb_channels=nb_channels,
            hidden_size=hidden_size,
            nb_layers=nb_layers,
            unidirectional=unidirectional,
            input_mean=input_mean,
            input_scale=input_scale,
            max_bin=max_bin,
        )

    def inference(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
