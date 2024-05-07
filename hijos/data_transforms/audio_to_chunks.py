from math import ceil

import numpy as np
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class AudioToChunks(DataTransform):
    def __init__(self, chunk_length: int, hop_length: int, stereo: bool = True) -> None:
        super().__init__()

        self.chunk_length = chunk_length
        self.hop_length = hop_length
        self.stereo = stereo

    def transform(self, input: np.ndarray, target: np.ndarray):
        if len(input) > self.chunk_length:
            num_chunks = ceil(len(input) / (self.hop_length))
            output = np.zeros((num_chunks, self.chunk_length, 2 if self.stereo else 1), dtype=np.float32)
            target_output = np.zeros(
                (num_chunks, target.shape[0], self.chunk_length, 2 if self.stereo else 1), dtype=np.float32
            )
            for i in range(num_chunks):
                index = i * self.hop_length
                input_chunk = input[index : index + self.chunk_length]
                target_chunk = target[:, index : index + self.chunk_length]
                output[i, : len(input_chunk)] = input_chunk
                target_output[i, :, : len(input_chunk)] = target_chunk
        else:
            output = np.zeros((1, self.chunk_length, 2 if self.stereo else 1), dtype=np.float32)
            target_output = np.zeros(
                (1, target.shape[0], self.chunk_length, 2 if self.stereo else 1), dtype=np.float32
            )
            output[0, : len(input)] = input
            target_output[0, :, : len(input)] = target
        return output, target_output
