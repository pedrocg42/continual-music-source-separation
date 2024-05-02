from typing import Literal

import torch
from bs_roformer import BSRoformer
from madre import register
from madre.extra.torch.models.torch_base_model import TorchBaseModel


@register()
class TorchBSRoformer(TorchBaseModel):
    def __init__(
        self,
        dim: int,
        depth: int,
        time_transformer_depth: int,
        freq_transformer_depth: int,
        num_stems: int,
        stereo: bool,
        stft_n_fft: int,
        window_sizes: tuple[int],
        window_fn: Literal["hann"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        match window_fn:
            case "hann":
                multi_stft_window_fn = torch.hann_window

        self.model = BSRoformer(
            dim=dim,
            depth=depth,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            num_stems=num_stems,
            stereo=stereo,
            stft_n_fft=stft_n_fft,
            multi_stft_resolutions_window_sizes=tuple(window_sizes),
            multi_stft_window_fn=multi_stft_window_fn,
        )

    def inference(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
