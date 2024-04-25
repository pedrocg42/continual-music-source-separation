import os

import musdb
import numpy as np
from madre.base.container.register import register
from madre.base.data.dataset.dataset import Dataset
from musdb.audio_classes import MultiTrack
from torch import Tensor

STEMS = ["mixture", "drums", "bass", "accompaniment", "vocals"]


@register()
class MUSDB18(Dataset):
    def __init__(
        self,
        name: str = "musdb18",
        targets: list[str] | None = None,
        split: str | None = None,
        is_wav: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name=name, **kwargs)

        self.targets = targets
        self.split = split

        self.split2subset = {
            "train": "train",
            "val": "train",
            "test": "test",
            None: ["train", "test"],
        }

        self.mus = musdb.DB(
            root=os.path.join(self.base_path, self.name),
            is_wav=is_wav,
            subsets=self.split2subset[self.split],
            split=self.split,
        )
        self.position = 0

    def __iter__(self) -> "MUSDB18":
        return self

    def __len__(self) -> int:
        return len(self.mus)

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.position >= len(self):
            self.position = 0
            raise StopIteration

        track: MultiTrack = self.mus[self.position]
        mixture = track.audio

        if self.targets is None:
            # returning all of them
            target_audios = track.stems[1:]
        else:
            target_audios = None
            for target in self.targets:
                if target_audios is None:
                    target_audios = track.targets[target].audio[np.newaxis]
                else:
                    target_audios = np.vstack([target_audios, track.targets[target].audio[np.newaxis]])

        return mixture, target_audios


@register()
class MUSDB18HQ(MUSDB18):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="musdb18hq", is_wav=True, **kwargs)
