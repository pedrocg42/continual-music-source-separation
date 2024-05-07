import numpy as np
from madre import register
from madre.evaluation.metrics.metric import Metric
from mir_eval import separation
from torch import Tensor


@register()
class MusicSourceSeparationMetric(Metric):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.metrics = {"SDR": []}

    def calculate(self, inputs: Tensor, targets: Tensor) -> None:
        breakpoint()
        inputs = inputs.numpy().reshape(-1, inputs.shape[-1])
        targets = targets.numpy().reshape(-1, targets.shape[-1])
        sdr = separation.bss_eval_sources(targets, inputs)[0].mean()
        self.metrics["SDR"].append(sdr)

    def log(self, epoch: int | None = None) -> None:
        for metric_name, metric in self.metrics.items():
            self.experiment_tracker.log_metric(metric_name, np.mean(metric), step=epoch)
