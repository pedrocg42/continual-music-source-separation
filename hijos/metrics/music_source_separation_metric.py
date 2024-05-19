import numpy as np
from einops import rearrange
from madre import register
from madre.evaluation.metrics.metric import Metric
from museval import evaluate

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
        self.metrics: dict[str, list[float]] = {
            "SDR": [],
            **{f"SDR_{target}": [] for target in self.targets},
            "ISR": [],
            **{f"ISR_{target}": [] for target in self.targets},
            "SIR": [],
            **{f"SIR_{target}": [] for target in self.targets},
            "SAR": [],
            **{f"SAR_{target}": [] for target in self.targets},
        }

    def calculate(self, batch_inputs: np.ndarray, batch_targets: np.ndarray) -> None:
        if batch_inputs.ndim == 3:
            # add batch dimension
            batch_inputs = batch_inputs[np.newaxis]
            batch_targets = batch_targets[np.newaxis]

        for inputs, targets in zip(batch_inputs, batch_targets, strict=True):
            inputs = rearrange(inputs, "nsrc nchan nsampl -> nsrc nsampl nchan")
            targets = rearrange(targets, "nsrc nchan nsampl -> nsrc nsampl nchan")
            sdr, isr, sir, sar = evaluate(targets, inputs)
            for i, (sdr_stem, isr_stem, sir_stem, sar_stem) in enumerate(
                zip(sdr, isr, sir, sar, strict=True)
            ):
                self.metrics[f"SDR_{self.targets[i]}"].append(np.nanmean(sdr_stem))
                self.metrics[f"ISR_{self.targets[i]}"].append(np.nanmean(isr_stem))
                self.metrics[f"SIR_{self.targets[i]}"].append(np.nanmean(sir_stem))
                self.metrics[f"SAR_{self.targets[i]}"].append(np.nanmean(sar_stem))
            self.metrics["SDR"].append(np.nanmean(sdr))
            self.metrics["ISR"].append(np.nanmean(isr))
            self.metrics["SDR"].append(np.nanmean(sir))
            self.metrics["ISR"].append(np.nanmean(sar))

    def log(self, epoch: int | None = None) -> None:
        for metric_name, metric in self.metrics.items():
            self.experiment_tracker.log_metric(metric_name, np.mean(metric), step=epoch)
        self._init_metrics()
