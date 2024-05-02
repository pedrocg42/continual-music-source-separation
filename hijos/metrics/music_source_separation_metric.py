from madre import register
from madre.evaluation.metrics.metric import Metric
from mir_eval import separation
from torch import Tensor


@register()
class MusicSourceSeparationMetric(Metric):
    def calculate(self, input: Tensor, target: Tensor) -> None:
        self.metrics["sdr"] = separation.bss_eval_sources(target, input)[0].mean()

    def log(self) -> None:
        self.experiment_tracker.log_metric(self.metrics["sdr"])
