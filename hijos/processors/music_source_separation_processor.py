import math

import torch
from madre.base.container.register import register
from madre.extra.torch.evaluation.torch_model_processor import TorchModelProcessor
from tqdm import tqdm


@register()
class MusicSourceSeparationProcessor(TorchModelProcessor):
    def __init__(self, eval_batch_size: int = 8, **kwargs) -> None:
        super().__init__(**kwargs)

        self.eval_batch_size = eval_batch_size

    @torch.no_grad()
    def execute(self, batch_inputs: torch.Tensor) -> torch.Tensor:
        inputs = batch_inputs[0]
        predictions = []
        iterator = tqdm(
            inputs.split(self.eval_batch_size),
            total=math.ceil(len(inputs) / self.eval_batch_size),
            desc="Inference by batches",
            colour="blue",
        )
        for subbatch_input in iterator:
            subbatch_input = subbatch_input.to(self.device, non_blocking=True)
            subbatch_predictions = self.model(subbatch_input)
            predictions.append(subbatch_predictions.cpu())
        return torch.vstack(predictions)[None]
