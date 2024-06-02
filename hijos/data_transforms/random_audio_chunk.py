import numpy as np
from madre import register
from madre.base.data.data_transforms.data_transform import DataTransform


@register()
class RandomAudioChunk(DataTransform):
    def __init__(self, chunk_length: int = 441000, stereo: bool = True) -> None:
        super().__init__()

        self.chunk_length = chunk_length
        self.stereo = stereo

    def transform(self, audio: np.ndarray, target: np.ndarray):
        len_audio = audio.shape[-1]
        if len_audio > self.chunk_length:
            random_index = np.random.randint(0, len_audio - self.chunk_length)
            output = audio[..., random_index : random_index + self.chunk_length]
            target_output = target[..., random_index : random_index + self.chunk_length]
        else:
            output = np.zeros((2 if self.stereo else 1, self.chunk_length))
            target_output = np.zeros((target.shape[0], 2 if self.stereo else 1, self.chunk_length))
            output[..., len_audio] = audio
            target_output[..., :len_audio] = target
        return output, target_output
