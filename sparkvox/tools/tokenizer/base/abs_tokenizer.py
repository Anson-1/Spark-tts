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
This file contains the abstract tokenizer class.
Other tokenizers should inherit this class.
"""


import torch


class ABSTokenizer(torch.nn.Module):
    """
    Abstract tokenizer class. Other tokenizers should inherit this class.
    """

    def tokenize(self, x):
        raise NotImplementedError

    def tokenize_batch(self, x, lengths=None):
        raise NotImplementedError

    def detokenize(self, x):
        raise NotImplementedError