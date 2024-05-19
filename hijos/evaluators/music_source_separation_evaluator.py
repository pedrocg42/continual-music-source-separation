import numpy as np
from einops import rearrange
from madre.base.container.register import register
from madre.evaluation.evaluators.evaluator import Evaluator
from torch import Tensor


@register()
class MusicSourceSeparationEvaluator(Evaluator):
    def __init__(
        self, chunk_length: int, hop_length: int, stereo: bool = True, apply_softmax: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.chunk_length = chunk_length
        self.hop_length = hop_length
        self.stereo = stereo
        self.apply_softmax = apply_softmax

    def execute(self, predictions: Tensor, targets: Tensor) -> tuple[np.ndarray, np.ndarray]:
        predictions = rearrange(predictions.cpu().numpy(), "(b s) stems 1 l -> b stems l s", s=2)
        predictions = rearrange(predictions, "b stems l s -> b stems s l")
        targets = rearrange(targets.numpy(), "(b s) stems 1 l -> b stems l s", s=2)
        targets = rearrange(targets, "b stems l s -> b stems s l")
        num_chunks, num_stems, num_channels, _ = targets.shape
        reconstructed_prediction = np.zeros(
            (num_stems, num_channels, self.hop_length * num_chunks + self.chunk_length - self.hop_length),
            dtype=np.float32,
        )
        reconstructed_target = np.zeros(
            (num_stems, num_channels, self.hop_length * num_chunks + self.chunk_length - self.hop_length),
            dtype=np.float32,
        )
        for i, (pred, y) in enumerate(zip(predictions, targets, strict=True)):
            if i == 0:
                window = self.trapzoid_window_function(start=False)
            elif (num_chunks + 1) == num_chunks:
                window = self.trapzoid_window_function(end=False)
            else:
                window = self.trapzoid_window_function()
            pred *= window[: pred.shape[-1]]
            start = i * self.hop_length
            reconstructed_prediction[:, :, start : start + pred.shape[-1]] = pred
            reconstructed_target[:, :, start : start + self.chunk_length] = y
        return reconstructed_prediction, reconstructed_target

    def trapzoid_window_function(self, start: bool = True, end: bool = True) -> np.ndarray:
        slope_length = self.chunk_length - self.hop_length
        slope = np.arange(slope_length) / slope_length
        window = np.ones(self.chunk_length)
        if start:
            window[:slope_length] = slope
        if end:
            window[-slope_length:] = slope[::-1]
        return window
