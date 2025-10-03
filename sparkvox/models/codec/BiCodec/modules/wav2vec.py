# Copyright (c) 2024 Xinsheng Wang (w.xinshawn@gmail.com)
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

# Adapted from https://github.com/descriptinc/descript-audio-codec under the Apache License 2.0

from typing import List

import math
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model


@torch.no_grad()
class Wav2Vec(nn.Module):
    def __init__(
        self,
        model_dir: str = '/aifs4su/xinshengwang/model/wav2vec2-large-xlsr-53',
        **kwargs
    ):
        super().__init__()

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(model_dir).eval()
        self.feature_extractor.config.output_hidden_states = True

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def forward(self, wavs: List[np.ndarray], device):
        self.feature_extractor.to(device)
        inputs = self.processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True, output_hidden_states=True).input_values.to(device)
        feat = self.feature_extractor(inputs)
        feats_16 = feat.hidden_states[16]
        feats_14 = feat.hidden_states[14]
        feats_11 = feat.hidden_states[11]
        feats_mix = (feats_11 + feats_14 + feats_16) / 3
        return feats_mix  #.half().detach().float()
       