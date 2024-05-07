from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from madre import register
from madre.extra.torch import DEVICE
from madre.extra.torch.train.criterias.torch_criteria import TorchCriteria
from torch import Tensor, stft


@register()
class TorchBsRoformerCriteria(TorchCriteria):
    def __init__(
        self,
        num_stems: int,
        resolution_weight: float,
        stft_n_fft: int,
        window_sizes: tuple[int],
        window_fn: Literal["hann"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_stems = num_stems
        self.resolution_weight = resolution_weight
        self.window_sizes = window_sizes
        self.stft_n_fft = stft_n_fft

        match window_fn:
            case "hann":
                self.window_fn = torch.hann_window

    def forward(self, recon_audio: Tensor, target: Tensor) -> dict[str, Tensor]:
        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")

        target = target[..., : recon_audio.shape[-1]]  # protect against lost length on istft

        audio_loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.0
        for window_size in self.window_sizes:
            res_stft_kwargs = {
                "n_fft": max(
                    window_size, self.stft_n_fft
                ),  # not sure what n_fft is across multi resolution stft
                "win_length": window_size,
                "return_complex": True,
                "window": self.window_fn(window_size, device=DEVICE),
                "hop_length": 147,
                "normalized": False,
            }

            recon_Y = stft(rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs)
            target_Y = stft(rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.resolution_weight

        total_loss = audio_loss + weighted_multi_resolution_loss

        return {
            "loss": total_loss,
            "audio_loss": audio_loss,
            "multi_resolution_loss": multi_stft_resolution_loss,
        }
