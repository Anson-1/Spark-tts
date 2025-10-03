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
import random
import numpy as np
import torch.nn.functional as F

from typing import Dict, List, Any
from omegaconf import DictConfig
from pathlib import Path

from sparkvox.utils.audio import load_audio
from sparkvox.utils.audio import audio_highpass_filter
from sparkvox.models.base.dataloaders.multi_jsonl_dataset import BaseDataset


class WavDataset(BaseDataset):
    """Dataset for wav and ref wav."""

    def __init__(self, config: DictConfig, mode: str = "val", extract_feat: bool = False, data_root: Path = None,  **kwargs) -> None:
        """
        Initialize the dataset with specific configuration and mode.

        Args:
            config (DictConfig): Dataset configuration as a dictionary.
            mode (str, optional): Specifies the mode, 'train' or 'val'. Defaults to 'train'.
        """
        super().__init__(config, mode, extract_feat, data_root)
        pass

    def get_sample(self, meta: DictConfig) -> Dict[str, Any]:
        """Get a sample from the metadata."""
        config = self.config
        index = meta["index"]
        if "wav_path_obs" in meta:
            wav_dir = meta["wav_path_obs"] if os.path.isabs(meta["wav_path_obs"]) else meta['wav_path']
        else:
            wav_dir = meta["wav_path"]

        if not os.path.isabs(wav_dir) and self.data_root is not None:
            wav_dir = os.path.join(self.data_root, wav_dir)

        try:
            wav = load_audio(wav_dir, config["sample_rate_for_ssl"], volume_normalize=True)
            ref_wav = load_audio(wav_dir, config["sample_rate"], volume_normalize=True)
            
            # highpass filter
            if config["highpass_cutoff_freq"] != 0:
                wav = audio_highpass_filter(
                    wav, config["sample_rate_for_ssl"], config["highpass_cutoff_freq"]
                )
                ref_wav = audio_highpass_filter(
                    ref_wav, config["sample_rate"], config["highpass_cutoff_freq"]
                )

            wav_length = len(ref_wav)
            ref_wav_length = config["ref_segment_duration"] * config["sample_rate"]

            # Repeat and truncate to handle insufficient length
            if wav_length < ref_wav_length:
                repeat_times = 1 + ref_wav_length // wav_length
                ref_wav = np.tile(ref_wav, repeat_times)

            ref_wav = ref_wav[:ref_wav_length]

            return {
                "index": index,
                "ref_wav": torch.from_numpy(ref_wav).float(),
                "wav": torch.from_numpy(wav).float(),
                "length": torch.tensor(wav_length),
            }

        except Exception as e:
            print("Bad case in fetch_data", e)
            return {
                "index": index,
                "ref_wav": None,
                "wav": None,
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

        collate_batch["index"] = [b["index"] for b in batch if b['wav'] is not None]
        collate_batch["wav"] = [b["wav"].numpy() for b in batch if b['wav'] is not None]
        collate_batch["length"] = torch.stack([b["length"] for b in batch if b['wav'] is not None], dim=0)
        collate_batch["ref_wav"] = torch.stack([b["ref_wav"] for b in batch if b['wav'] is not None], dim=0)

        return collate_batch


# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = {
        "datasets": {
            "highpass_cutoff_freq": 40,
            "sample_rate": 16000,
            "segment_duration": 2.4,
            "max_val_duration": 12,
            "latent_hop_length": 320,
            "ref_segment_duration": 6,
            "jsonlfiles": {
                "train": "/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test.jsonl",
                "val": "/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test.jsonl",
            },
        },
    }

    dataset = WavDataset(config["datasets"])
    dataloader = DataLoader(
        dataset, batch_size=12, num_workers=4, collate_fn=dataset.collate_fn
    )

    i = 0
    for batch in dataloader:
        i += 1
        print(f"itr {i}, wav_size:", batch["wav"].shape)
