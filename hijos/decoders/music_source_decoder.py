import torch
from madre import register
from madre.base.data.decoder import Decoder


def collate_fun(batch: tuple[torch.Tensor, torch.Tensor]):
    inputs = []
    targets = []
    for x, y in batch:
        inputs.append(x)
        targets.append(y)
    return torch.vstack(inputs), torch.vstack(targets)


@register()
class MusicSourceDecoder(Decoder):
    def __init__(self, stereo_to_batch: bool = True):
        self.stereo_to_batch = stereo_to_batch

    @staticmethod
    def get_collate_function():
        return collate_fun
