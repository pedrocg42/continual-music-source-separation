import os
import musdb
from musdb.audio_classes import MultiTrack
from torch.utils.data import Dataset
from torch import Tensor

from config import datasets_path

STEMS = ["mixture", "drums", "bass", "accompaniment", "vocals"]


class MUSDB18(Dataset):
    def __init__(self, name: str = "musdb18", stem: str | None = None) -> None:
        super().__init__()

        self.name = name
        self.stem = stem

        self.mus = musdb.DB(root=os.path.join(datasets_path, self.name))
        self.position = 0

        self.stem2index = {stem: i for i, stem in enumerate(STEMS)}

    def __len__(self) -> int:
        return len(self.mus)

    def __next__(self) -> tuple[Tensor, Tensor]:
        track: MultiTrack = self.mus[self.position]
        mixture = track.audio

        if self.stem is None:
            target = track.stems[1:]
        else:
            target = track.stems[self.stem2index[self.stem]]

        self.position += 1
        if self.position >= len(self):
            self.position = 0

        return mixture, target


class MUSDB18HQ(Dataset):
    def __init__(self, name: str = "musdb18hq", *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)
