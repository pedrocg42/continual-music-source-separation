import math

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from analysis.utils.audio import audio_to_chunks


class Separator:
    def __init__(
        self,
        model_path: str,
        sources_names: list[str],
        batch_length: int = 1,
        chunk_length: int = 44100 * 10,
        hop_length: int = 44100 * 10,
    ) -> None:
        self.model_path = model_path
        self.sources_names = sources_names
        self.batch_length = batch_length

        # audio processing
        self.chunk_length = chunk_length
        self.hop_length = hop_length

        self.model = self.load_model()

    def load_model(self) -> torch.nn.Module:
        with open(self.model_path, "rb") as fp:
            model = torch.load(fp, map_location="cpu")
        model.cuda()
        model.eval()
        return model

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.separate(*args, **kwargs)

    @torch.no_grad()
    def separate(self, audio: np.ndarray) -> np.ndarray:
        chunks = audio_to_chunks(audio, chunk_length=self.chunk_length, hop_length=self.hop_length)
        predicted_sources = self.predict_batches(chunks)
        sources = rearrange(predicted_sources, "batch source channel length -> source channel (batch length)")
        return sources

    @torch.no_grad()
    def predict_batches(self, chunks: torch.Tensor) -> tuple[torch.Tensor, float]:
        predictions = []
        total_batches = math.ceil(len(chunks) / self.batch_length)
        for i in tqdm(range(total_batches), desc="Inference by batches", colour="blue"):
            batch_input = chunks[i * self.batch_length : (i + 1) * self.batch_length]
            batch_input = torch.as_tensor(batch_input).cuda()
            batch_predictions = self.model(batch_input)
            predictions.append(batch_predictions.cpu().numpy())
        return np.concatenate(predictions, axis=-1)
