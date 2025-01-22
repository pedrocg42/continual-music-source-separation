import os

import h5py
import musdb
import numpy as np
from loguru import logger
from madre.base.container.register import register
from madre.base.data.dataset.dataset import Dataset
from madre.base.data.dataset.dataset_item import DatasetItem
from musdb.audio_classes import MultiTrack
from tqdm import tqdm

STEMS = ["mixture", "vocals", "drums", "bass", "other"]
TARGETS = ["vocals", "drums", "bass", "other", "accompaniment"]


@register()
class MUSDB18(Dataset):
    def __init__(
        self,
        name: str = "musdb18",
        base_path: str = os.getenv("DATASETS_PATH", "."),
        targets: list[str] | str = "all",
        subsets: list[str] | None = None,
        split: str | None = None,
        is_wav: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(name=name, base_path=base_path, **kwargs)

        if isinstance(targets, str):
            if targets == "all":
                self.targets = TARGETS[:4]
            else:
                self.targets = [targets]
        else:
            self.targets = targets
        self.subsets = subsets
        self.split = split

        self.dataset_path = os.path.join(self.base_path, self.name)
        self.h5_folder = os.path.join(self.dataset_path, "h5")
        os.makedirs(self.h5_folder, exist_ok=True)

        self.mus = musdb.DB(root=self.dataset_path, is_wav=is_wav, subsets=self.subsets, split=self.split)
        self.parse()

    def parse(self) -> None:
        logger.info(f"Parse {self.__class__.__name__}")
        for idx in tqdm(range(len(self.mus))):
            track: MultiTrack = self.mus[idx]

            file_path = os.path.join(self.h5_folder, f"{track.name}.h5")
            self.items[track.name] = DatasetItem(id=track.name, data=file_path)

            if os.path.isfile(file_path):
                continue

            with h5py.File(file_path, mode="w") as hf:
                data = track.audio.astype(np.float32)
                hf.create_dataset("mixture", data=data)
                for target in TARGETS:
                    data = track.targets[target].audio.astype(np.float32)
                    hf.create_dataset(target, data=data)

    def __next__(self) -> DatasetItem:
        item = super().__next__()

        with h5py.File(item.data, mode="r") as hf:
            mixture = np.asarray(hf.get("mixture"))

            targets = np.zeros((len(self.targets), len(mixture), 2), dtype=np.float32)
            for i, target_name in enumerate(self.targets):
                targets[i] = np.asarray(hf.get(target_name))[np.newaxis]

        return DatasetItem(id=item.id, data=mixture, target=targets)


@register()
class MUSDB18HQ(MUSDB18):
    def __init__(self, **kwargs) -> None:
        super().__init__(name="musdb18hq", is_wav=True, **kwargs)
