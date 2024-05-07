import torch
from madre import register
from madre.base.data.decoder import Decoder


def collate_fun(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for x, y in batch:
        inputs.append(x)
        targets.append(y)
    x = torch.vstack(inputs)
    y = torch.vstack(targets)
    return x, y


@register()
class MusicSourceDecoder(Decoder):
    def __init__(self, stereo_to_batch: bool = True):
        self.stereo_to_batch = stereo_to_batch

    def get_collate_function(self):
        if self.stereo_to_batch:
            return collate_fun
        return None
