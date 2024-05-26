import math

import numpy as np
import torch
from madre.base.container.register import register
from madre.extra.torch.train.loopers.torch_looper import TorchLooper
from tqdm import tqdm


@register()
class MusicSourceSeparationLooper(TorchLooper):
    def __init__(self, eval_batch_size: int, **kwargs):
        super().__init__(**kwargs)

        self.eval_batch_size = eval_batch_size

    @torch.no_grad()
    def eval_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.eval()

        iterator = tqdm(
            self.eval_iterator,
            total=self.max_steps if self.max_steps else self.eval_batch_count,
            desc="👩🏽‍🔬 Evaluation",
            colour="magenta",
        )
        batch_results = {"loss": []}
        for i, (batch_inputs, batch_targets) in enumerate(iterator):
            for inputs, targets in zip(batch_inputs, batch_targets, strict=True):
                predictions, loss = self.eval_batch(inputs, targets)
                batch_results["loss"].append(loss)
                predictions, targets = self.evaluator(predictions, targets)
                for metric in self.metrics:
                    metric(predictions, targets)
            if self.max_steps and (i + 1) >= self.max_steps:
                break
        eval_loss = np.mean(batch_results["loss"])
        self.experiment_tracker.log_metric("Eval.Loss", eval_loss, step=epoch)
        for metric in self.metrics:
            metric.log(epoch)
        return {"eval_loss": eval_loss}

    @torch.no_grad()
    def eval_batch(
        self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        predictions = []
        loss = []

        iterator = tqdm(
            zip(
                batch_inputs.split(self.eval_batch_size),
                batch_targets.split(self.eval_batch_size),
                strict=True,
            ),
            total=math.ceil(len(batch_inputs) / self.eval_batch_size),
            desc="Inference Song by Batches",
            colour="blue",
        )
        for x_subbatch, y_subbatch in iterator:
            x_subbatch = x_subbatch.to(self.device, non_blocking=True)
            y_subbatch = y_subbatch.to(self.device, non_blocking=True)
            batch_predictions = self.model(x_subbatch)
            loss.append(self.criteria(batch_predictions, y_subbatch)["loss"].item())
            predictions.append(batch_predictions.cpu())
        return torch.vstack(predictions), np.mean(loss)
