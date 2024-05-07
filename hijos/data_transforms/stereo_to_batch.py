import numpy as np
from einops import rearrange
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class StereoToBatch(DataTransform):
    def transform(self, inputs: np.ndarray, targets: np.ndarray):
        if inputs.ndim == 2:
            return inputs.T[:, np.newaxis], np.transpose(targets, (2, 0, 1))[:, :, np.newaxis]
        elif inputs.ndim == 3:
            inputs = rearrange(inputs, "b l s -> (b s) 1 l")
            targets = rearrange(targets, "b stems l s -> (b s) stems 1 l")
            return inputs, targets
        else:
            raise ValueError("Wrong number of dimensions")
