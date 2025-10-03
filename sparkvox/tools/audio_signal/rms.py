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
Calculate energy features from audio signal
"""

import numpy as np
import librosa
from typing import Tuple

def calculate_rms(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # Use same parameters as pitch calculation for consistency
    hop_length = int(sample_rate * 0.005)
    win_length = hop_length * 6
    
    # Calculate RMS energy
    energy = librosa.feature.rms(
        y=audio,
        frame_length=win_length,
        hop_length=hop_length,
        center=True
    )
    
    # Convert to 1D array
    energy = energy.squeeze()
    # Remove any potential invalid values
    energy[energy < 0] = 0
    energy[energy == np.nan] = 0
    
    return energy

def normalize_rms(rms: np.ndarray) -> np.ndarray:
    rms = rms[rms > 0]
    max_rms = np.max(rms)
    min_rms = np.min(rms)
    norm_rms = (rms - min_rms) / (max_rms - min_rms)
    return norm_rms

def get_rms_mean_and_std(audio: np.ndarray, sample_rate: int) -> Tuple[float, float]:
    rms = calculate_rms(audio, sample_rate)
    rms = rms[rms > 0]
    mean = np.mean(rms)
    norm_rms = normalize_rms(rms)
    norm_std = np.std(norm_rms)
    return mean, norm_std

if __name__ == "__main__":
    from sparkvox.utils.audio import load_audio
    # audio, sample_rate = librosa.load("/aifs4su/xinshengwang/code/spark-tts/sparkvox/local/wav.wav", sr=16000)
    audio, sample_rate = librosa.load("/aifs4su/mmdata/rawdata/speech/Expresso/wavs/ex04_whisper_00380.wav", sr=16000)
    audio2 = load_audio("/aifs4su/mmdata/rawdata/speech/Expresso/wavs/ex04_whisper_00380.wav", sampling_rate=16000, volume_normalize=True)
    # rms = calculate_rms(audio, sample_rate)
    # rms2 = calculate_rms(audio2, sample_rate)
    rms_mean, rms_std = get_rms_mean_and_std(audio, sample_rate)
    rms_mean2, rms_std2 = get_rms_mean_and_std(audio2, sample_rate)
    print(rms_mean, rms_std)
    print(rms_mean2, rms_std2)