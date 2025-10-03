#!/bin/bash

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


# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

# Set default parameters
audio_tokenizer_config="egs/speech_synthesis/spark-tts/config/audio_tokenizer/bicodec.yaml"
ckpt="/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/logs/20250925_165457/ckpt/last.ckpt"
# ckpt="/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/Spark-TTS-0.5B/LLM/model.safetensors"
save_dir='local/sparktts/infer/sparktts_qwen0.5b/20250922_135844'
wav_path="egs/data/audios/m3ed/m3ed_Neutral_0000022926.wav"
text="靠你这张脸买东西能打折。"
# prompt_text='我怕你万一累出个三长两短来。'
device=1

# Change directory to the root directory
cd "$root_dir" || exit

source sparkvox/utils/parse_options.sh

# Run inference for each JSON file
python -m sparkvox.models.speech_synthesis.sparktts.inference_single \
    --ckpt "${ckpt}" \
    --audio_tokenizer_config "${audio_tokenizer_config}" \
    --save_dir "${save_dir}" \
    --wav_path "${wav_path}" \
    --text "${text}" \
    --device "${device}" \
    --prompt_text "${prompt_text}"