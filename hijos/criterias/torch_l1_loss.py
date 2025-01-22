import torch.nn.functional as F
from madre import register
from madre.extra.torch.train.criterias.torch_criteria import TorchCriteria
from torch import Tensor


@register()
class TorchSourceSeparationL1Criteria(TorchCriteria):
    def forward(self, recon_audio: Tensor, target: Tensor) -> dict[str, Tensor]:
        dims = tuple(range(2, recon_audio.dim()))
        loss = F.l1_loss(recon_audio, target, reduction="none")
        loss = loss.mean(dims).mean(1).mean(0)
        return {"loss": loss}
