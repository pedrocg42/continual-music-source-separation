import numpy as np
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class RandomAudioChunk(DataTransform):
    def __init__(self, chunk_length: int = 352800, stereo: bool = True) -> None:
        super().__init__()

        self.chunk_length = chunk_length
        self.stereo = stereo

    def transform(self, input: np.ndarray, target: np.ndarray):
        if len(input) > self.chunk_length:
            random_index = np.random.randint(0, len(input) - self.chunk_length)
            output = input[random_index : random_index + self.chunk_length]
            target_output = target[:, random_index : random_index + self.chunk_length]
        else:
            output = np.zeros((self.chunk_length, 2 if self.stereo else 1))
            target_output = np.zeros((target.shape[0], self.chunk_length, 2 if self.stereo else 1))
            output[: len(input)] = input
            target_output[:, : len(input)] = target
        return output, target_output
