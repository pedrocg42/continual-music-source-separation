import random

import torch
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform
from torchaudio.transforms import PitchShift


@register()
class PitchShiftTransform(DataTransform):
    def __init__(
        self, sample_rate: int = 44100, min_shift: int = -4, max_shift: int = 4, p: float = 0.5
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.p = p

        self.transforms = [
            PitchShift(sample_rate=sample_rate, n_steps=n) for n in range(min_shift, max_shift + 1)
        ]

    @torch.no_grad()
    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        if random.uniform(0, 1) < self.p:
            transform = self.transforms[random.randint(self.min_shift, self.max_shift)]
            audio = transform(audio)
            target = transform(target)
        return audio, target
