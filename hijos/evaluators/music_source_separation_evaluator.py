import numpy as np
from einops import rearrange
from madre.base.container.register import register
from madre.base.data.dataset.dataset_item import DatasetItem
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

    def execute(
        self,
        dataset_item: DatasetItem,
        **kwargs,
    ):
        """
            Compute non-genuine probability for the input features dict items.
        :param features_dict: Dict[str, Any]
            Features dict following the format below:
                * { "frame_key": Any}
                    or,
                * {"frame_hey": {"features_key": Any}
        :return: float
            Score!
        """
        if len(dataset_item.features) > 1:
            raise RuntimeError(
                "BW Evaluator Error: Found features for several frames, while evaluator expects only one"
            )
        frame_features = list(dataset_item.features.values())[0]
        _, feature = next(iter(frame_features.items()))
        preds = self.softmax(feature) if self.apply_softmax else feature
        dataset_item.score = 1.0 - preds[0]

    def execute_batch(self, predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        breakpoint()
        predictions = rearrange(predictions.numpy(), "(b s) stems 1 l -> b stems s l", s=2)
        targets = rearrange(targets.numpy(), "(b s) stems 1 l -> b stems s l", s=2)
        num_chunks, num_stems, num_channels, chunk_length = targets.shape
        reconstructed_prediction = np.zeros(
            (num_stems, num_channels, self.hop_length * num_chunks), dtype=np.float32
        )
        reconstructed_target = np.zeros(
            (num_stems, num_channels, chunk_length * num_chunks), dtype=np.float32
        )
        breakpoint()
        for i, (pred, y) in enumerate(zip(predictions, targets, strict=True)):
            if i == 0:
                window = self.trapzoid_window_function(start=False)
            elif (num_chunks + 1) == num_chunks:
                window = self.trapzoid_window_function(end=False)
            else:
                window = self.trapzoid_window_function()
            pred *= window[: len(pred)]
            start = i * self.hop_length
            end = start + self.chunk_length
            reconstructed_prediction[:, :, start:end] = pred
            reconstructed_target[:, :, start:end] = y
        breakpoint()
        return predictions, targets

    def trapzoid_window_function(self, start: bool = True, end: bool = True) -> np.ndarray:
        slope_length = self.chunk_length - self.hop_length
        slope = np.arange(slope_length) / slope_length
        window = np.ones(self.chunk_length)
        if start:
            window[:slope_length] = slope
        if end:
            window[-slope_length:] = slope[::-1]
        return window
