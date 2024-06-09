import random

import torch
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform
from torchaudio.transforms import InverseSpectrogram, Spectrogram, TimeStretch


@register()
class TimeStretchTransform(DataTransform):
    def __init__(
        self, sample_rate: int = 44100, min_stretch: float = 0.8, max_stretch: float = 1.2, p: float = 0.5
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch
        self.p = p

        self.spec = Spectrogram(power=None)
        self.stretch = TimeStretch()
        self.ispec = InverseSpectrogram()

    @torch.no_grad()
    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        if random.uniform(0, 1) < self.p:
            stretch = round(random.uniform(self.min_stretch, self.max_stretch), ndigits=2)
            audio_spec = self.spec(audio)
            audio_spec_stretched = self.stretch(audio_spec, stretch)
            audio = self.ispec(audio_spec_stretched)
            target_spec = self.spec(target)
            target_spec_stretched = self.stretch(target_spec, stretch)
            target = self.ispec(target_spec_stretched)
        return audio, target
