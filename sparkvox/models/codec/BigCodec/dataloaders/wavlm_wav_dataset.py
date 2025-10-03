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


import torch
import random
import numpy as np
import torch.nn.functional as F

from typing import Dict, List, Any
from omegaconf import DictConfig

from sparkvox.utils.audio import load_audio
from sparkvox.utils.audio import audio_highpass_filter
from sparkvox.models.base.dataloaders.multi_jsonl_dataset import BaseDataset


class WavlmWavCodecDataset(BaseDataset):
    """Dataset for wavlm to wav codec."""

    def __init__(self, config: DictConfig, mode: str = "train", **kwargs) -> None:
        """
        Initialize the dataset with specific configuration and mode.

        Args:
            config (DictConfig): Dataset configuration as a dictionary.
            mode (str, optional): Specifies the mode, 'train' or 'val'. Defaults to 'train'.
        """
        super().__init__(config, mode)
        pass

    def get_sample(self, meta: DictConfig) -> Dict[str, Any]:
        """Get a sample from the metadata."""
        config = self.config
        index = meta["index"]
        wav_dir = meta["wav_path_obs"]
        feat_dir = config["feat_dir"]

        try:
            feat = torch.load(f"{feat_dir}/{index}.pt")
            length = feat.shape[0]
            wav_length = length * config["latent_hop_length"]
            wav = load_audio(
                wav_dir, config["sample_rate"], volume_normalize=True, length=wav_length
            )

            length = length // 4 * 4
            wav_length = length * config["latent_hop_length"]
            feat = feat[:length]
            wav = wav[:wav_length]

            if config["highpass_cutoff_freq"] != 0:
                wav = audio_highpass_filter(
                    wav, config["sample_rate"], config["highpass_cutoff_freq"]
                )

            if not self.train:
                duration = wav_length // config["sample_rate"]
                config["segment_duration"] = min(duration, config["max_val_duration"])

            segment_length = (
                (
                    int(config["sample_rate"] * config["segment_duration"])
                    // config["latent_hop_length"]
                )
                // 4
                * 4
            )
            wav_segment_length = segment_length * config["latent_hop_length"]

            if wav_segment_length > wav_length:
                wav = np.pad(wav, (0, int(wav_segment_length - wav_length)))
                feat = F.pad(
                    feat, (0, 0, 0, segment_length - length), mode="constant", value=0
                )
                start_indice = 0

            else:
                if not self.train:
                    start_indice = 0
                else:
                    start_indice = random.randint(0, length - segment_length)

            end_indice = start_indice + segment_length
            wav_start_indice = start_indice * config["latent_hop_length"]
            wav_end_indice = end_indice * config["latent_hop_length"]
            segment = feat[start_indice:end_indice]
            wav_segment = wav[wav_start_indice:wav_end_indice]

            wav = torch.from_numpy(wav)
            wav_length = len(wav)
            length = feat.shape[-1]
            ref_wav_length = config["ref_segment_duration"] * config["sample_rate"]

            mask = torch.ones_like(wav)

            if length / segment_length > 2 and self.train:
                mask[wav_start_indice:wav_end_indice] = 0

            ref_wav = wav * mask

            if wav_length < ref_wav_length:
                repeat_times = 1 + ref_wav_length // wav_length
                ref_wav = ref_wav.repeat(repeat_times)

            ref_length = len(ref_wav)
            ref_start_indices = random.randint(0, ref_length - ref_wav_length)
            ref_end_indices = ref_start_indices + ref_wav_length
            ref_segment_wav = ref_wav[ref_start_indices:ref_end_indices]

            return {
                "index": index,
                "feat": segment.float(),
                "ref_wav": ref_segment_wav.float(),
                "wav": torch.from_numpy(wav_segment).float(),
            }

        except Exception as e:
            print("Bad case in fetch_data", e)
            return {
                "index": index,
                "feat": None,
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

        collate_batch["index"] = [b["index"] for b in batch if b["wav"] is not None]
        for k in ("feat", "wav", "ref_wav"):
            v = [b[k] for b in batch if b["feat"] is not None]
            if v[0] is not None:
                collate_batch[k] = torch.stack(v, dim=0)
            else:
                collate_batch[k] = None

        return collate_batch


# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = {
        "sample_rate": 16000,
        "latent_hop_length": 320,
        "highpass_cutoff_freq": 40,
        "segment_duration": 2.4,  # (s)
        "max_val_duration": 12,
        "ref_segment_duration": 6,
        "feat_dir": "/aifs4su/xinshengwang/data/speech/vocoder/wav2vec/wav2vec_mix",
        "jsonlfiles": {
            "train": "/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test.jsonl",
            "val": "/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test.jsonl",
        },
    }

    dataset = WavlmWavCodecDataset(config)
    dataloader = DataLoader(
        dataset, batch_size=12, num_workers=4, collate_fn=dataset.collate_fn
    )

    i = 0
    for batch in dataloader:
        i += 1
        print(f"itr {i}, wav_size:", batch["wav"].shape)
