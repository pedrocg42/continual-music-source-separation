import math

import numpy as np
import torch
from tqdm import tqdm

from analysis.utils.audio import audio_to_chunks, concatenate_chunks


class Separator:
    def __init__(
        self,
        model_path: str,
        sources_names: list[str],
        batch_length: int = 8,
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
        model.eval()
        return model

    def __call__(self, *args, **kwargs) -> np.ndarray:
        self.separate(*args, **kwargs)

    @torch.no_grad()
    def separate(self, audio: np.ndarray) -> np.ndarray:
        breakpoint()
        chunks = audio_to_chunks(audio, chunk_length=self.chunk_length, hop_length=self.hop_length)
        predicted_chunks = self.predict_batches(torch.as_tensor(chunks))
        sources = np.asarray(concatenate_chunks(predicted_chunks))
        return sources

    @torch.no_grad()
    def predict_batches(self, chunks: torch.Tensor) -> tuple[torch.Tensor, float]:
        predictions = []
        iterator = tqdm(
            chunks.split(self.batch_length),
            total=math.ceil(len(chunks) / self.batch_length),
            desc="Inference by batches",
            colour="blue",
        )
        for batch_input in iterator:
            batch_predictions = self.model(batch_input)
            predictions.append(batch_predictions.cpu())
        return torch.vstack(predictions)
