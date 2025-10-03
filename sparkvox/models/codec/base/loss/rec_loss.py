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

# Taken from https://github.com/Aria-K-Alethia/BigCodec/blob/main/criterions/mel_loss.py


import torch
import torch.nn as nn

from typing import List
from torchaudio.transforms import MelSpectrogram


class MultiResolutionMelSpectrogramLoss(nn.Module):
    """Multi-resolution mel spectrogram loss."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
        mel_fmin: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
    ):
        """Initialize the multi-resolution mel spectrogram loss.

        Args:
            sample_rate (int): Sample rate.
            n_mels (List[int]): Number of mels per STFT.
            window_lengths (List[int]): Length of each window of each STFT.
            clamp_eps (float): Clamp on the log magnitude, below.
            pow (float): Power to raise magnitude to before taking log.
        """
        super().__init__()
        self.mel_transforms = nn.ModuleList(
            [
                MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=n_mel,
                    power=1.0,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                )
                for n_mel, window_length in zip(n_mels, window_lengths)
            ]
        )
        self.n_mels = n_mels
        self.loss_fn = nn.L1Loss()
        self.clamp_eps = clamp_eps
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for mel_transform in self.mel_transforms:
            x_mel = mel_transform(x)
            y_mel = mel_transform(y)
            log_x_mel = x_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_y_mel = y_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            loss += self.loss_fn(log_x_mel, log_y_mel)
        return loss
