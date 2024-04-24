import torch
from bs_roformer import BSRoformer

import config
from hijos.data_transforms.random_audio_chunk import RandomAudioChunk
from hijos.data_transforms.stereo_to_batch import StereoToBatch
from hijos.data_transforms.to_torch_tensor import ToTorchTensor
from hijos.datasets.musdb18 import MUSDB18HQ

torch.set_grad_enabled(False)

model = BSRoformer(
    dim=512, depth=12, time_transformer_depth=1, freq_transformer_depth=1, num_stems=1
)
model.train()

dataset_train = MUSDB18HQ(
    base_path=config.datasets_path,
    targets=["vocals", "drums", "bass", "other"],
    split="train",
)
data_transforms = [
    RandomAudioChunk(chunk_length=44100 * 8),
    StereoToBatch(),
    ToTorchTensor(),
]

model.train()
model.to(config.device)
for x, y in dataset_train:

    for data_transform in data_transforms:
        x, y = data_transform(x, y)

    x = x.to(config.device)
    y = y.to(config.device)

    out = model(x)


print("End")
