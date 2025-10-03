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
Calculate Perceived loudness from audio signal
"""

import pyloudnorm as pyln

import numpy as np


def calculate_loudness(audio: np.ndarray, sample_rate: int):
    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(audio, -1.0)
    # measure the loudness first 
    meter = pyln.Meter(sample_rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(peak_normalized_audio)
    return loudness


if __name__ == "__main__":
    import time
    from sparkvox.utils.audio import load_audio
    audio_file1 = "/aifs4su/mmdata/rawdata/speech/Expresso/wavs/ex04_whisper_00380.wav"
    audio_file2 = "/aifs4su/xinshengwang/code/spark-tts/sparkvox/local/wav.wav"
    audio1 = load_audio(audio_file1, sampling_rate=16000, volume_normalize=True)
    audio2 = load_audio(audio_file2, sampling_rate=16000, volume_normalize=True)

    start = time.time()
    loudness_value1 = calculate_loudness(audio1, 16000)
    loudness_value2 = calculate_loudness(audio2, 16000)
    end = time.time()
    print('time', end - start)

    print(f"Perceived loudness: {loudness_value1}")
    print(f"Perceived loudness: {loudness_value2}")
