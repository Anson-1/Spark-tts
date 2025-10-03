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


import os
import torch
import numpy as np

from pathlib import Path
from typing import Dict, List, Any
from omegaconf import DictConfig
from torch.utils.data import Dataset

from sparkvox.utils.audio import load_audio
from sparkvox.models.base.dataloaders.multi_jsonl_dataset import BaseDataset

class GenderDataset(BaseDataset):
    """Dataset for gender classification."""
    def __init__(self, config: DictConfig, mode: str = "train", **kwargs) -> None:
        """
        Initialize the dataset with specific configuration and mode.

        Args:
            config (DictConfig): Dataset configuration as a dictionary.
            mode (str, optional): Specifies the mode, 'train' or 'val'. Defaults to 'train'.
        """
        super().__init__(config, mode)
        
        self.segment_length = config["audio_duration"] * config["sample_rate"]
        self.gender_map = {
            'f': 0, 
            'm': 1,
            'female': 0,
            'male': 1,
        }

    def get_sample(self, meta: DictConfig) -> Dict[str, Any]:
        """Get a sample from the metadata."""
        index = meta['index']
        gender = self.gender_map[meta['gender'].lower()]
        wav = load_audio(meta['wav_path'], sampling_rate=self.config["sample_rate"])
        wav_length = len(wav)
        wav = torch.from_numpy(wav)
        # repeat the ref wav if it is shorter than the segment length
        if self.segment_length > wav_length:
            repeat_times = 1 + self.segment_length // wav_length
            wav = wav.repeat(repeat_times)
        
        wav = wav[: self.segment_length]
        length = len(wav)

        return {
            'index': index,
            'wav': wav,
            'length': torch.tensor(length),
            'labels': torch.tensor(gender)
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

        max_length = max([b['length'] for b in batch])
        collate_batch['wav'] = torch.zeros(len(batch), max_length).float()
    
        collate_batch['index'] = [b['index'] for b in batch]
        collate_batch['length'] = torch.stack([b['length'] for b in batch], dim=0)
        collate_batch['labels'] = torch.stack([b['labels'] for b in batch], dim=0)

        for i, sample in enumerate(batch):
            wav = sample['wav']
            wav_length = sample['length']
            collate_batch['wav'][i, :wav_length] = wav
            
        return collate_batch

# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    config = {
        "datasets": {
            "jsonlfiles":{
            "train": "/home/xinshengwang/data/sparkvox/tmp/vctk/test.jsonl",
            "val": "/home/xinshengwang/data/sparkvox/tmp/vctk/test.jsonl",
            },
        },
    }

    dataset = GenderDataset(config)
    dataloader = DataLoader(dataset, batch_size=12, num_workers=4,
                        collate_fn=dataset.collate_fn)
    
    i = 0
    for batch in dataloader:
        i += 1
        print(f"itr {i}, wav_size:", batch['wav'].shape)