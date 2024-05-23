from typing import Literal

from bs_roformer import BSRoformer
from madre import register
from madre.extra.torch.models.torch_base_model import TorchBaseModel
from torch import Tensor, hann_window


@register()
class TorchBSRoformer(TorchBaseModel):
    def __init__(
        self,
        dim: int,
        depth: int,
        time_transformer_depth: int,
        freq_transformer_depth: int,
        heads: int,
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
                multi_stft_window_fn = hann_window

        self.model = BSRoformer(
            dim=dim,
            depth=depth,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            heads=heads,
            num_stems=num_stems,
            stereo=stereo,
            stft_n_fft=stft_n_fft,
            multi_stft_resolutions_window_sizes=tuple(window_sizes),
            multi_stft_window_fn=multi_stft_window_fn,
        )

    def inference(self, audio: Tensor) -> Tensor:
        normalized_audio, mean, std = self.normalize(audio)
        separated_audio = self.model(normalized_audio)
        denormalized_separated_audio = self.denormalize(separated_audio, mean, std)
        return denormalized_separated_audio

    @staticmethod
    def normalize(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mono = x.mean(dim=1, keepdim=True)
        mean = mono.mean(dim=-1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        x = (x - mean) / (1e-5 + std)
        return x, mean, std

    @staticmethod
    def denormalize(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        return x * std + mean
