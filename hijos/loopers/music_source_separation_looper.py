import numpy as np
import torch
from madre.base.container.register import register
from madre.extra.torch.train.loopers.torch_looper import TorchLooper
from tqdm import tqdm


@register()
class MusicSourceSeparationLooper(TorchLooper):
    def __init__(self, batch_size: int, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size

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
        for metric in self.metrics:
            metric.log()
        return {"eval_loss": np.mean(batch_results["loss"])}

    @torch.no_grad()
    def eval_batch(
        self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        predictions = []
        loss = []
        for x_subbatch, y_subbatch in zip(
            batch_inputs.split(self.batch_size), batch_targets.split(self.batch_size), strict=True
        ):
            batch_predictions = self.model(x_subbatch)
            predictions.append(batch_predictions)
            loss.append(self.criteria(batch_predictions, y_subbatch)["loss"].item())
        return torch.vstack(predictions), np.mean(loss)
