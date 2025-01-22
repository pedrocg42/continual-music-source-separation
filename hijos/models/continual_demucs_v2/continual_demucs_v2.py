from demucs.demucs import Demucs
from torch import nn


class ContinualDemucs(Demucs):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

        self.embeddings = nn.Parameter()
