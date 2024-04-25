import torch
from bs_roformer import BSRoformer
from madre.models.torch_base_model import TorchBaseModel


class TorchBSRoformer(TorchBaseModel):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.model = BSRoformer(dim=512, depth=12, time_transformer_depth=1, freq_transformer_depth=1)

    def inference(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
