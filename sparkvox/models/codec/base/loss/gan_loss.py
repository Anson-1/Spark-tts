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
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class GANLoss(nn.Module):
    """GAN loss."""

    def __init__(self):
        super(GANLoss, self).__init__()

    def disc_loss(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        real_loss = F.mse_loss(real, torch.ones_like(real))
        fake_loss = F.mse_loss(fake, torch.zeros_like(fake))
        return real_loss, fake_loss

    def gen_loss(self, fake: torch.Tensor) -> torch.Tensor:
        gen_loss = F.mse_loss(fake, torch.ones_like(fake))
        return gen_loss
