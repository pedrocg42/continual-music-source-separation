import numpy as np
import torch
from madre.base.data.data_transforms.data_transform import DataTransform


class ToTorchTensor(DataTransform):
    def transform(self, input: np.ndarray, target: np.ndarray):
        return torch.as_tensor(input, dtype=torch.float32), torch.as_tensor(target, dtype=torch.float32)
