import numpy as np
import torch
from madre import register
from madre.evaluation.metrics.metric import Metric
from torch import Tensor

STEMS = ["mixture", "vocals", "drums", "bass", "other"]


@register()
class MusicSourceSeparationMetric(Metric):
    def __init__(self, targets: list[str] | str = "all", **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(targets, str):
            if targets == "all":
                self.targets = STEMS[1:]
            else:
                self.targets = [targets]
        else:
            self.targets = targets
        self._init_metrics()

    def _init_metrics(self) -> None:
        self.metrics: dict[str, list[float]] = {"SDR": [], **{f"SDR_{target}": [] for target in self.targets}}

    def calculate(self, batch_preds: Tensor, batch_targets: Tensor) -> None:
        if batch_preds.ndim == 3:
            # add batch dimension
            batch_preds = batch_preds[None]
            batch_targets = batch_targets[None]
        elif batch_preds.ndim < 3:
            raise RuntimeError("Not enough dimensions")

        sdr = MusicSourceSeparationMetric.calculare_sdr_demucs(
            batch_targets[..., : batch_preds.shape[-1]], batch_preds
        ).numpy()
        self.metrics["SDR"].append(np.mean(sdr))
        for sdr_target, target_name in zip(sdr, self.targets, strict=True):
            self.metrics[f"SDR_{target_name}"].append(np.mean(sdr_target))

    def log(self, epoch: int | None = None) -> None:
        for metric_name, metric in self.metrics.items():
            self.experiment_tracker.log_metric(metric_name, np.mean(metric), step=epoch)
        self._init_metrics()

    @staticmethod
    def calculare_sdr_demucs(references: Tensor, estimates: Tensor) -> Tensor:
        """
        Compute the SDR according to demucs repository
        https://github.com/facebookresearch/demucs
        """
        assert references.dim() == 4
        assert estimates.dim() == 4
        delta = 1e-7  # avoid numerical errors
        num = torch.sum(torch.square(references), dim=(2, 3)) + delta
        den = torch.sum(torch.square(references - estimates), dim=(2, 3)) + delta
        scores = 10 * torch.log10(num / den)
        return scores.T

    @staticmethod
    def calculare_sdr_bytedance(reference: np.ndarray, estimates: np.ndarray) -> np.ndarray:
        """
        Compute the SDR according to bytedance repository
        https://github.com/bytedance/music_source_separation
        """
        s_true = reference
        s_artif = estimates - reference
        sdr = 10.0 * (
            np.log10(np.clip(np.mean(s_true**2), 1e-8, np.inf))
            - np.log10(np.clip(np.mean(s_artif**2), 1e-8, np.inf))
        )
        return sdr
