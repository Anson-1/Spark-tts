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
Discriminator module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict, Any, List
from audiotools import AudioSignal

from sparkvox.models.codec.base.loss import GANLoss
from sparkvox.models.codec.base.modules.wave_discriminator_dac import (
    WaveDiscriminator as wav_discriminator,
)


class Discriminator(nn.Module):
    """Discriminator module."""

    def __init__(
        self, sample_rate: int, wav_discriminator: nn.Module, **kwargs
    ) -> None:
        super().__init__()
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()
        self.wav_discriminator = wav_discriminator
        self.sample_rate = sample_rate

    def forward(
        self, fake: AudioSignal, real: AudioSignal
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        d_fake = self.wav_discriminator(fake.audio_data)
        d_real = self.wav_discriminator(real.audio_data)
        return d_fake, d_real

    def adversarial_loss(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get adversarial loss

        Args:
            inputs (dict): A dictionary that should contains the following elemetns:
                recons (torch.Tensor): Synthetic audios with shape [B, T]
                audios (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict): A dictionary that contains the following elements:
                adv_loss (torch.Tensor): Adversarial loss
                feature_map_loss (torch.Tensor): Feature map loss
        """
        
        signal = AudioSignal(inputs["audios"].clone(), self.sample_rate)
        recons = AudioSignal(inputs["recons"].clone(), signal.sample_rate)

        d_fake, d_real = self.forward(recons, signal)

        adv_loss = 0
        for x_fake in d_fake:
            adv_loss += self.gan_loss.gen_loss(x_fake[-1])

        feature_map_loss = 0
        
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                feature_map_loss += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
       
        return {"adv_loss": adv_loss, "feature_map_loss": feature_map_loss}

    def discriminative_loss(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Get discriminative loss

        Args:
            inputs (Dict[str, Any]): A dictionary that should contains the following elemetns:
                recons (torch.Tensor): Synthetic audios with shape [B, T]
                audios (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            disc_loss (torch.Tensor): Discriminative loss
        """
        signal = AudioSignal(inputs["audios"], self.sample_rate)
        recons = AudioSignal(inputs["recons"], signal.sample_rate)
        disc_loss = self.compute_discriminator_loss(recons, signal)

        return disc_loss

    def compute_discriminator_loss(
        self, fake: AudioSignal, real: AudioSignal
    ) -> torch.Tensor:
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)

        return loss_d

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


# test
if __name__ == "__main__":
    model = Discriminator(
        sample_rate=16000, wav_discriminator=wav_discriminator(sample_rate=16000)
    )
    x = torch.zeros(4, 1, 16000)
    disc_loss = model.discriminative_loss(inputs={"recons": x, "audios": x})
    adv_loss = model.adversarial_loss(inputs={"recons": x, "audios": x})
    print(disc_loss, adv_loss)
