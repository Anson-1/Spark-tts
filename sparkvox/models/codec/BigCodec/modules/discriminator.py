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

from typing import Tuple, Dict, Any
from hydra.utils import instantiate
from omegaconf import DictConfig

from sparkvox.models.codec.base.loss import GANLoss
from sparkvox.models.codec.BigCodec.modules.mdp import HiFiGANMultiPeriodDiscriminator as wav_discriminator
from sparkvox.models.codec.BigCodec.modules.mstft import SpecDiscriminator as spec_discriminator

class Discriminator(nn.Module):
    """Discriminator module."""

    def __init__(
        self,
        wav_discriminator: nn.Module,
        spec_discriminator: nn.Module,
        **kwargs
    ) -> None:
        super().__init__()
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()
        self.wav_discriminator = wav_discriminator
        self.spec_discriminator = spec_discriminator

    def forward(
        self, fake: torch.Tensor, real: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_fake = self.wav_discriminator(fake)
        d_fake_spec = self.spec_discriminator(fake)
        d_real, d_real_spec = None, None
        if real is not None:
            d_real = self.wav_discriminator(real)
            d_real_spec = self.spec_discriminator(real)

        return d_real, d_fake, d_real_spec, d_fake_spec

    def adversarial_loss(self, recons: torch.Tensor) -> torch.Tensor:
        """
        Get adversarial loss

        Args:
            inputs (dict): A dictionary that should contains the following elemetns:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'loss'
        """

        _, d_fake, _, d_fake_spec = self.forward(recons)

        adv_loss_list = []
        for x_fake in d_fake:
            adv_loss = self.gan_loss.gen_loss(x_fake[-1])
            adv_loss_list.append(adv_loss)

        for x_fake in d_fake_spec:
            adv_loss = self.gan_loss.gen_loss(x_fake[-1])
            adv_loss_list.append(adv_loss)

        adv_loss = sum(adv_loss_list)

        return adv_loss

    def discriminative_loss(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fake, real = inputs["recons"], inputs["audios"]
        d_real, d_fake, d_real_spec, d_fake_spec = self.forward(
            fake.clone().detach(), real
        )

        real_loss_list, fake_loss_list = [], []

        for x_fake, x_real in zip(d_fake, d_real):
            real_loss, fake_loss = self.gan_loss.disc_loss(x_real[-1], x_fake[-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        for x_fake, x_real in zip(d_fake_spec, d_real_spec):
            real_loss, fake_loss = self.gan_loss.disc_loss(x_real[-1], x_fake[-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def feature_match_loss(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        fake, real = inputs["recons"], inputs["audios"]
        d_real, d_fake, d_real_spec, d_fake_spec = self.forward(fake, real)

        fm_loss = 0.0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                fm_loss += self.l1_loss(d_fake[i][j], d_real[i][j].detach())

        return fm_loss

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
