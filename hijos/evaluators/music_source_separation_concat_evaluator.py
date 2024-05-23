import numpy as np
from einops import rearrange
from madre.base.container.register import register
from madre.evaluation.evaluators.evaluator import Evaluator
from torch import Tensor


@register()
class MusicSourceSeparationConcatEvaluator(Evaluator):
    def execute(self, predictions: Tensor, targets: Tensor) -> tuple[np.ndarray, np.ndarray]:
        predictions = rearrange(predictions.cpu(), "b s c l -> s c (b l)")
        targets = rearrange(targets, "b s c l -> s c (b l)")
        return predictions, targets
