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
Gender predictor with pytorch lightning
"""

import torch
import torch.nn as nn
import torchaudio.transforms as TT

from hydra.utils import instantiate
from typing import Dict, Any
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from sparkvox.models.speaker_attribute.base.lightning_models.mel_classifier import (
    MelClassifier,
)


class GenderPredictor(MelClassifier):
    """Gender predictor."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        super().__init__(config)
        pass


# test
if __name__ == "__main__":
    from sparkvox.utils.file import load_config

    config = load_config(
        "egs/speaker_attribute/gender/config/mel_gender_predictor.yaml"
    )
    model = GenderPredictor(config["model"])
    wav = torch.zeros([8, 16000])
    label = torch.zeros(8)
    batch = {"wav": wav, "labels": label}
    output = model(batch)
    print(output["logits"])
