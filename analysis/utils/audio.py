import io
import math

import librosa
import numpy as np
import streamlit as st
import torch
from einops import rearrange
from matplotlib import pyplot as plt


def plot_audio(audio: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(audio, linewidth=0.04)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    st.pyplot(fig)


@st.cache_data(show_spinner=True)
def read_audio(
    file: str | io.BytesIO, sample_rate: int = 44100, stereo: bool = True, plot: bool = False
) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file, sr=sample_rate)
    if stereo and (len(audio.shape) == 1 or audio.shape[0] == 1):
        audio = np.repeat(audio[:, None] if len(audio.shape) == 1 else audio, repeats=2, axis=1)

    if plot:
        plot_audio(audio)
    return audio, sr


def audio_to_chunks(audio: np.ndarray, chunk_length: int, hop_length: int) -> np.ndarray:
    num_chunks = math.ceil(len(audio) / (hop_length))
    output = np.zeros((num_chunks, chunk_length, 2), dtype=np.float32)
    for i in range(num_chunks):
        index = i * hop_length
        input_chunk = audio[index : index + chunk_length]
        output[i, : len(input_chunk)] = input_chunk
    return output


def concatenate_chunks(chunks: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return rearrange(chunks, "b s c l -> s c (b l)")
