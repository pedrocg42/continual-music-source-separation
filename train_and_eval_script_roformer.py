import os

import torch
from bs_roformer import BSRoformer
from madre.base.data.data_decoder import DataDecoder
from madre.extra.torch.data.torch_data_loader import TorchDataLoader

import config
from hijos.criterias.torch_bs_roformer_criteria import TorchBsRoformerCriteria
from hijos.data_transforms.random_audio_chunk import RandomAudioChunk
from hijos.data_transforms.stereo_to_batch import StereoToBatch
from hijos.data_transforms.to_torch_tensor import ToTorchTensor
from hijos.datasets.musdb18 import MUSDB18HQ

torch.set_grad_enabled(False)

# Configs
targets = ["vocals", "drums", "bass", "other"]
num_stems = len(targets)
multi_stft_resolution_loss_weight = 1.0
multi_stft_resolutions_window_sizes = (4096, 2048, 1024, 512, 256)
stft_n_fft = 2048
window_fn = "hann"

criteria = TorchBsRoformerCriteria(
    num_stems=num_stems,
    resolution_weight=multi_stft_resolution_loss_weight,
    window_sizes=multi_stft_resolutions_window_sizes,
    window_fn=window_fn,
    stft_n_fft=stft_n_fft,
)
criteria.to(config.device)

model = BSRoformer(
    dim=512,
    depth=3,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_stems=num_stems,
    stereo=True,
    stft_n_fft=stft_n_fft,
    multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
    multi_stft_window_fn=window_fn,
)
model.to(config.device)

dataset_train = MUSDB18HQ(
    base_path=os.getenv("DATASETS_PATH", "."),
    targets=["vocals", "drums", "bass", "other"],
    split="train",
)
data_transforms = [
    RandomAudioChunk(chunk_length=44100 * 8),
    StereoToBatch(),
    ToTorchTensor(),
]
decoder = DataDecoder(data_transforms=data_transforms)

data_loader = TorchDataLoader(
    dataset=dataset_train,
    data_decoder=decoder,
    batch_size=8,
    num_workers=0,
    collate_function=decoder.get_collate_function(),
)

# Train
model.train()
for x, target in data_loader:
    x = x.to(config.device)
    target = target.to(config.device)

    recon_audio = model(x)

    loss = criteria(recon_audio, target)

    loss["loss"]


print("End")
