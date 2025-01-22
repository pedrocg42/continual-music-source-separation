import torch
from einops import rearrange
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class Rearrange(DataTransform):
    """
    Rearrange tensors
    """

    def __init__(self, input_pattern: str, target_pattern: str) -> None:
        super().__init__()

        self.input_pattern = input_pattern
        self.target_pattern = target_pattern

    def transform(self, inputs: torch.Tensor, target: torch.Tensor):
        inputs = rearrange(inputs, self.input_pattern)
        target = rearrange(target, self.target_pattern)
        return inputs, target
