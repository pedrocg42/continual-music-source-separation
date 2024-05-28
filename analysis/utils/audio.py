import io
import math

import librosa
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from scipy.io import wavfile


def plot_audio(audio: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(audio.T, linewidth=0.04)
    ax.set_axis_off()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    st.pyplot(fig)


@st.cache_data(show_spinner=True)
def read_audio(
    file: str | io.BytesIO, sample_rate: int = 44100, stereo: bool = True, plot: bool = False
) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(file, sr=sample_rate, mono=not stereo)
    if stereo and (len(audio.shape) == 1 or audio.shape[0] == 1):
        audio = np.repeat(audio[None, :] if len(audio.shape) == 1 else audio, repeats=2, axis=0)

    if plot:
        plot_audio(audio)
    return audio, sr


def write_audio(filename: str | io.BytesIO, audio: np.ndarray, sample_rate: int = 44100) -> None:
    wavfile.write(filename, rate=sample_rate, data=audio)


def audio_to_chunks(audio: np.ndarray, chunk_length: int, hop_length: int) -> np.ndarray:
    num_chunks = math.ceil(audio.shape[1] / (hop_length))
    output = np.zeros((num_chunks, 2, chunk_length), dtype=np.float32)
    for i in range(num_chunks):
        index = i * hop_length
        input_chunk = audio[:, index : index + chunk_length]
        output[i, :, : input_chunk.shape[1]] = input_chunk
    return output
