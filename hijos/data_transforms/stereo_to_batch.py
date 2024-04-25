import numpy as np
from madre.base.data.data_transforms.data_transform import DataTransform


class StereoToBatch(DataTransform):
    def transform(self, input: np.ndarray, target: np.ndarray):
        return input.T, np.transpose(target, (0, 2, 1))
