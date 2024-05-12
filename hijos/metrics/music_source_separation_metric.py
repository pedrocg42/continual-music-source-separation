import numpy as np
from madre import register
from madre.evaluation.metrics.metric import Metric
from mir_eval import separation
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

    def _init_metrics(self):
        self.metrics: dict[str, list[float]] = {"SDR": [], **{f"SDR_{target}": [] for target in self.targets}}

    def calculate(self, batch_inputs: Tensor, batch_targets: Tensor) -> None:
        if batch_inputs.ndim == 3:
            # add batch dimension
            batch_inputs = batch_inputs[np.newaxis]
            batch_targets = batch_targets[np.newaxis]
        for inputs, targets in zip(batch_inputs, batch_targets, strict=True):
            sdr_sources = []
            for i, (input, target) in enumerate(zip(inputs, targets, strict=True)):
                # 0 -> SDR, 1 -> SIR, 2 -> SAR
                sdr = separation.bss_eval_sources(target, input)[0].mean()
                self.metrics[f"SDR_{self.targets[i]}"].append(sdr)
            self.metrics["SDR"].append(np.mean(sdr_sources))

    def log(self, epoch: int | None = None) -> None:
        for metric_name, metric in self.metrics.items():
            self.experiment_tracker.log_metric(metric_name, np.mean(metric), step=epoch)
        self._init_metrics()
