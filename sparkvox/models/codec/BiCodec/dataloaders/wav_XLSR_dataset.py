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
import soxr
import numpy as np

from pathlib import Path
from typing import Dict, List, Any, Tuple
from omegaconf import DictConfig

from sparkvox.utils.audio import load_audio
from sparkvox.utils.audio import audio_highpass_filter
from sparkvox.models.base.dataloaders.multi_jsonl_dataset import BaseDataset


class BicodecDataset(BaseDataset):
    """Dataset for wav2vec + mel -> wav codec."""

    def __init__(self, config: DictConfig, mode: str = "train", **kwargs) -> None:
        """
        Initialize the dataset with specific configuration and mode.

        Args:
            config (DictConfig): Dataset configuration as a dictionary.
            mode (str, optional): Specifies the mode, 'train' or 'val'. Defaults to 'train'.
        """
        super().__init__(config, mode)
        pass

    def load_audio(self, wav_dir: Path) -> Tuple[np.ndarray, int]:
        # load the audio
        wav = load_audio(wav_dir, self.config["sample_rate"], volume_normalize=True)
        # make sure the length is divisible by 4
        # to support the possible downsampling in the model
        length = len(wav) // self.config["latent_hop_length"] // 4 * 4
        wav_length = length * self.config["latent_hop_length"]
        wav = wav[:wav_length]

        # highpass filter
        if self.config["highpass_cutoff_freq"] != 0:
            wav = audio_highpass_filter(
                wav, self.config["sample_rate"], self.config["highpass_cutoff_freq"]
            )
            
        return wav, wav_length

    def get_reference_segment(self, wav: np.ndarray, start_indice: int, end_indice: int) -> np.ndarray:
        """Get a reference segment from the wav."""
        wav_length = len(wav)
        wav = torch.from_numpy(wav)
        segment_length = self.config["segment_duration"] * self.config["sample_rate"]
        ref_wav_length = self.config["ref_segment_duration"] * self.config["sample_rate"]

        # mask the ref wav to avoid content leaking
        mask = torch.ones_like(wav)
        if wav_length / segment_length > 2 and self.train:
            mask[start_indice:end_indice] = 0
        ref_wav = wav * mask

        # repeat the ref wav if it is shorter than the ref length
        if wav_length < ref_wav_length:
            repeat_times = 1 + ref_wav_length // wav_length
            ref_wav = ref_wav.repeat(repeat_times)

        # random crop for training
        ref_length = len(ref_wav)
        ref_start_indices = random.randint(0, ref_length - ref_wav_length)
        ref_end_indices = ref_start_indices + ref_wav_length
        ref_segment_wav = ref_wav[ref_start_indices:ref_end_indices]

        return ref_segment_wav.float()
        
    def get_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Get a sample from the metadata.
    
        Args:
            meta (Dict[str, Any]): Metadata for the sample.
                - index (int): Index of the sample.
                - sample_rate (int): Sample rate for the audio.
                - latent_hop_length (int): Hop length for the latent features.
                - highpass_cutoff_freq (int): Cutoff frequency for the highpass filter.
                - segment_duration (float): Duration of the segment.
                - max_val_duration (float): Maximum duration for validation.
                - ref_segment_duration (float): Duration of the reference segment.
                - wav_path (Path): Path to the observed wav file.
                
        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - index (int): Index of the sample.
                - ref_wav (torch.Tensor): Reference waveform tensor.
                - wav (torch.Tensor): Waveform tensor.
        """
            
        config = self.config
        index = meta["index"]
        wav_dir = meta["wav_path"]
        sample_rate = config["sample_rate"]
        latent_hop_length = config["latent_hop_length"]
        segment_duration = config["segment_duration"]
        additonal_length = config["additonal_duration_per_side"] * 16000
        drop_frames = config["additonal_duration_per_side"] * 50 if self.train else 0
        
        try:
            wav, wav_length = self.load_audio(wav_dir)
            
            # long audio segment for validation
            if not self.train:
                duration = wav_length // sample_rate
                segment_duration = min(duration, config["max_val_duration"])

            segment_length = (int(sample_rate * segment_duration) // latent_hop_length) * latent_hop_length
            expected_num_frames = int(segment_length / sample_rate * 50)
            # pad the wav if it is shorter than the segment length
            if segment_length > wav_length:
                wav = np.pad(wav, (0, int(segment_length - wav_length)))
                start_indice = 0
            
            # random crop for training
            else:
                if not self.train:
                    start_indice = 0
                else:
                    start_indice = random.randint(0, wav_length - segment_length)
            
            start_indice = start_indice * sample_rate * 16000 // 16000 // sample_rate 
            
            end_indice = start_indice + segment_length
            
            # get target segment
            wav_segment_out = wav[start_indice:end_indice]
            # get the reference segment
            ref_segment_wav = self.get_reference_segment(wav, start_indice, end_indice)
            
            if sample_rate != 16000:
                wav = soxr.resample(wav, sample_rate, 16000, quality="VHQ")
                start_indice = start_indice * 16000 // sample_rate
                end_indice = end_indice * 16000 // sample_rate
                wav_length = len(wav)

            if self.train:
                # append additional length in both sides for ssl extraction
                adjusted_start_indice = start_indice - additonal_length
                adjusted_end_indice = wav_length - end_indice - additonal_length
                start_indice = start_indice - additonal_length
                end_indice = end_indice + additonal_length

                if adjusted_start_indice < 0:
                    wav = np.pad(wav, (abs(adjusted_start_indice), 0))
                    start_indice = 0
                    end_indice = end_indice + abs(adjusted_start_indice)
                if adjusted_end_indice < 0:
                    wav = np.pad(wav, (0, abs(adjusted_end_indice)))    
            wav_segment_in = wav[start_indice:end_indice]

            return {
                "index": index,
                "drop_frames": drop_frames,
                'expected_num_frames': expected_num_frames,
                "ref_wav": ref_segment_wav,
                "wav_in": torch.from_numpy(wav_segment_in).float().numpy(),
                "wav": torch.from_numpy(wav_segment_out).float(),
                }

        except Exception as e:
            print("Bad case in fetch_data", e)
            return {
                "index": index,
                "ref_wav": None,
                "wav_in": None,
                "wav_out": None,
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

        collate_batch['drop_frames'] = batch[0]['drop_frames']
        collate_batch['expected_num_frames'] = batch[0]['expected_num_frames']
        for k in ("index", "wav_in"):
            collate_batch[k] = [b[k] for b in batch]

        for k in ("wav", "ref_wav"):
            v = [b[k] for b in batch]
            if v[0] is not None:
                collate_batch[k] = torch.stack(v, dim=0)
            else:
                collate_batch[k] = None

        return collate_batch


# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = {
        "jsonlfiles": {
            "train": "/aifs4su/xinshengwang/data/voxbox/speech/Cantonese/PhoenixTV_subset_and_mix/metadata.jsonl",
            "val": "/aifs4su/xinshengwang/data/voxbox/speech/Cantonese/PhoenixTV_subset_and_mix/metadata.jsonl",
        },
        "highpass_cutoff_freq": 40,
        "sample_rate": 24000 ,
        "segment_duration": 2.4,
        "max_val_duration": 12,
        "latent_hop_length": 480,
        "ref_segment_duration": 6,
        "volume_normalize": True,
        "additonal_duration_per_side": 2,
    }

    dataset = BicodecDataset(config)
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=4, collate_fn=dataset.collate_fn
    )

    i = 0
    for batch in dataloader:
        i += 1
        print(f"itr {i}", "wav_in_size:",  batch["wav_in"][0].shape, "wav_out_size:", batch["wav"].shape, "ref_wav_size:", batch["ref_wav"].shape)
