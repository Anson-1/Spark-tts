# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dataset for age classification.
"""

import torch

from typing import Dict, List, Any
from omegaconf import DictConfig

from sparkvox.utils.audio import load_audio
from sparkvox.models.base.dataloaders.multi_jsonl_dataset import BaseDataset


class WavDataset(BaseDataset):
    """Dataset for speaker attribute prediction."""

    def __init__(self, config: DictConfig, mode: str = "val", extract_feat: bool = True, **kwargs) -> None:
        """
        Initialize the dataset with specific configuration and mode.

        Args:
            config (DictConfig): Dataset configuration as a dictionary.
            mode (str, optional): Specifies the mode, 'train' or 'val'. Defaults to 'train'.
        """
        super().__init__(config, mode, extract_feat=extract_feat)

        self.wav_length = config["audio_duration"] * config["sample_rate"]
        self.filter_metadata()
        

    def filter_metadata(self):
        metadata =[]
        for meta in self.metadata:
            index = meta["index"]
            wav = load_audio(meta["wav_path"], sampling_rate=self.config["sample_rate"])
            if len(wav) < 10:
                continue
            metadata.append(meta)
        self.metadata = metadata

    def get_sample(self, meta: DictConfig) -> Dict[str, Any]:
        """Get a sample from the metadata."""
        index = meta["index"]
        wav = load_audio(meta["wav_path"], sampling_rate=self.config["sample_rate"])
        wav = torch.from_numpy(wav)
        if len(wav) < self.wav_length:
            repeat_times = 1 + self.wav_length // len(wav)
            wav = wav.repeat(repeat_times)

        wav = wav[: self.wav_length]

        length = len(wav)

        return {
            "index": index,
            "wav": wav,
            "length": torch.tensor(length)
        }

    def collate_fn(self, batch: List[dict]) -> Dict[str, List]:
        """
        Collate function to pad batch data for training.

        Args:
            batch (List[Dict]): List of data dictionaries to collate.

        Returns:
            Dict[str, List]: Collated batch data.
        """
        assert isinstance(batch, list)
        collate_batch = {}

        max_length = max([b["length"] for b in batch])
        collate_batch["wav"] = torch.zeros(len(batch), max_length).float()

        collate_batch["index"] = [b["index"] for b in batch]
        collate_batch["length"] = torch.stack([b["length"] for b in batch], dim=0)

        for i, sample in enumerate(batch):
            wav = sample["wav"]
            wav_length = sample["length"]
            collate_batch["wav"][i, :wav_length] = wav

        return collate_batch


# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = {
        "sample_rate": 16000,
        "max_audio_duration": 10,
        "jsonlfiles": {
                "train": "/home/xinshengwang/data/sparkvox/tmp/vctk/test.jsonl",
                "val": "/home/xinshengwang/data/sparkvox/tmp/vctk/test.jsonl",
            },
        }

    dataset = WavDataset(config)
    dataloader = DataLoader(
        dataset, batch_size=12, num_workers=4, collate_fn=dataset.collate_fn
    )

    i = 0
    for batch in dataloader:
        i += 1
        print(f"itr {i}, wav_size:", batch["wav"].shape)
