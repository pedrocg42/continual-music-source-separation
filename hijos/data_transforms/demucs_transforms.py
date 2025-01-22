import random

import torch
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class FlipChannels(DataTransform):
    """
    Flip left-right channels.
    """

    def __init__(self, proba: float = 0.5):
        self.proba = proba

    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        if random.uniform(0, 1) < self.proba:
            audio = torch.flip(audio, dims=[0])
            target = torch.flip(target, dims=[1])
        return audio, target


@register()
class FlipSignTransform(DataTransform):
    def __init__(self, proba: float = 0.5):
        self.proba = proba

    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        if random.uniform(0, 1) < self.proba:
            audio = audio * -1
            target = target * -1
        return audio, target


@register()
class ScaleTransform(DataTransform):
    def __init__(self, proba: float = 1.0, min: float = 0.25, max: float = 1.25) -> None:
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max
        self.delta = self.max - self.min

    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        if random.uniform(0, 1) < self.proba:
            scale = torch.rand(1, device=audio.device, dtype=torch.float32)
            scale = scale * self.delta + self.min
            audio *= scale
            target *= scale
        return audio, target
