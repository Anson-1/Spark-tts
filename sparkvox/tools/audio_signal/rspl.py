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
Calculate RSPL features from audio signal
"""

import numpy as np
import librosa
from typing import Tuple
from sparkvox.tools.audio_signal.rms import calculate_rms, normalize_rms


def calculate_rspl(
    audio: np.ndarray, sample_rate: int, reference_pressure: float = 20e-6
) -> np.ndarray:
    rms = calculate_rms(audio, sample_rate)
    rms[rms<=0] = reference_pressure
    rspl = 20 * np.log10(rms / reference_pressure)

    return rspl


def get_rspl_mean_and_std(audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    rspl = calculate_rspl(audio, sample_rate)
    rspl = rspl[rspl > 0]
    mean = np.mean(rspl)
    norm_rspl = normalize_rms(rspl)
    norm_std = np.std(norm_rspl)
    return mean, norm_std


if __name__ == "__main__":
    audio, sample_rate = librosa.load(
        "/aifs4su/xinshengwang/code/spark-tts/sparkvox/local/wav.wav", sr=16000
    )
    rspl = calculate_rspl(audio, sample_rate)
    rspl_mean, rspl_std = get_rspl_mean_and_std(audio, sample_rate)
    print(rspl_mean, rspl_std)
