from collections.abc import Callable

import torch
import torch.nn.functional as F
from bs_roformer import BSRoformer
from einops import rearrange
from madre.base.data.data_decoder import DataDecoder
from madre.extra.torch.torch_data_loader import TorchDataLoader

import config
from hijos.data_transforms.random_audio_chunk import RandomAudioChunk
from hijos.data_transforms.stereo_to_batch import StereoToBatch
from hijos.data_transforms.to_torch_tensor import ToTorchTensor
from hijos.datasets.musdb18 import MUSDB18HQ

torch.set_grad_enabled(False)


def calculate_loss(
    num_stems: int,
    multi_stft_resolution_loss_weight: float,
    multi_stft_resolutions_window_sizes: tuple[int],
    stft_n_fft: int,
    window_fn: Callable,
    target: torch.Tensor,
    recon_audio: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if num_stems > 1:
        assert target.ndim == 4 and target.shape[1] == num_stems

    if target.ndim == 2:
        target = rearrange(target, "... t -> ... 1 t")

    target = target[..., : recon_audio.shape[-1]]  # protect against lost length on istft

    audio_loss = F.l1_loss(recon_audio, target)

    multi_stft_resolution_loss = 0.0
    for window_size in multi_stft_resolutions_window_sizes:
        res_stft_kwargs = {
            "n_fft": max(window_size, stft_n_fft),  # not sure what n_fft is across multi resolution stft
            "win_length": window_size,
            "return_complex": True,
            "window": window_fn(window_size, device=config.device),
            "hop_length": 147,
            "normalized": False,
        }

        recon_Y = torch.stft(rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs)
        target_Y = torch.stft(rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs)

        multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

    weighted_multi_resolution_loss = multi_stft_resolution_loss * multi_stft_resolution_loss_weight

    total_loss = audio_loss + weighted_multi_resolution_loss

    return {
        "loss": total_loss,
        "audio_loss": audio_loss,
        "multi_resolution_loss": multi_stft_resolution_loss,
    }


targets = ["vocals", "drums", "bass", "other"]
num_stems = len(targets)

multi_stft_resolution_loss_weight = 1.0
multi_stft_resolutions_window_sizes = (4096, 2048, 1024, 512, 256)
stft_n_fft = 2048
window_fn = torch.hann_window

model = BSRoformer(
    dim=512,
    depth=12,
    time_transformer_depth=1,
    freq_transformer_depth=1,
    num_stems=num_stems,
    stereo=True,
    stft_n_fft=stft_n_fft,
    multi_stft_resolutions_window_sizes=multi_stft_resolutions_window_sizes,
    multi_stft_window_fn=window_fn,
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
decoder = DataDecoder(data_transforms=data_transforms)

data_loader = TorchDataLoader(
    dataset=dataset_train,
    data_decoder=decoder,
    batch_size=8,
    num_workers=0,
    collate_function=decoder.get_collate_function(),
)


model.train()
model.to(config.device)


for x, target in data_loader:
    x = x.to(config.device)
    target = target.to(config.device)

    recon_audio = model(x)

    loss_dict = calculate_loss(
        num_stems,
        multi_stft_resolution_loss_weight,
        multi_stft_resolutions_window_sizes,
        stft_n_fft,
        window_fn,
        target,
        recon_audio,
    )

    print(loss_dict)


print("End")
