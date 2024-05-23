import torch
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class FlipChannels(DataTransform):
    """
    Flip left-right channels.
    """

    def forward(self, audio: torch.Tensor, target: torch.Tensor):
        batch, sources, channels, time = audio.size()
        target_batch, target_sources, target_channels, target_time = target.size()
        left = torch.randint(2, (batch, sources * target_sources, 1, 1), device=audio.device)
        left = left.expand(-1, -1, -1, time)
        right = 1 - left
        audio = torch.cat([audio.gather(2, left), audio.gather(2, right)], dim=2)
        return audio, target


@register()
class FlipSignTransform(DataTransform):
    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        batch, sources, channels, time = audio.size()
        target_batch, target_sources, target_channels, target_time = target.size()
        signs = torch.randint(
            2, (batch, sources + target_sources, 1, 1), device=audio.device, dtype=torch.float32
        )
        audio = audio * (2 * signs[:, 0] - 1)
        target = target * (2 * signs[:, 1:] - 1)
        return audio, target


@register()
class ScaleTransform(DataTransform):
    def __init__(self, proba: float = 1.0, min: float = 0.25, max: float = 1.25) -> None:
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def transform(self, audio: torch.Tensor, target: torch.Tensor):
        batch, sources, channels, time = audio.size()
        target_batch, target_sources, target_channels, target_time = target.size()
        scales = torch.empty(batch, sources + target_sources, 1, 1, device=audio.device).uniform_(
            self.min, self.max
        )
        audio *= scales[:, 0]
        target *= scales[:, 1:]
        return audio, target
