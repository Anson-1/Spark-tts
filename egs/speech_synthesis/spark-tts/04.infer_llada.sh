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
#ckpt="/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/LLADA_8B_Instruct"
ckpt="/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/logs_llada8b/20250902_164055/ckpt/epoch-92.ckpt"
save_dir='local/sparktts/infer/sparktts_llada/20250902_diffusion'
wav_path="egs/data/audios/m3ed/m3ed_Neutral_0000022926.wav"
text="靠你这张脸买东西能打折。"
prompt_text='我怕你万一累出个三长两短来。'
device=3
# LLaDA-specific parameters
steps=64
gen_length=512
block_length=32
cfg_scale=0.0
remasking="low_confidence"

# Change directory to the root directory
cd "$root_dir" || exit

source sparkvox/utils/parse_options.sh

# Run inference for each JSON file
python -m sparkvox.models.speech_synthesis.sparktts.inference_single_llada \
    --ckpt "${ckpt}" \
    --audio_tokenizer_config "${audio_tokenizer_config}" \
    --save_dir "${save_dir}" \
    --wav_path "${wav_path}" \
    --text "${text}" \
    --device "${device}" \
    --prompt_text "${prompt_text}" \
    --steps "${steps}" \
    --gen_length "${gen_length}" \
    --block_length "${block_length}" \
    --cfg_scale "${cfg_scale}" \
    --remasking "${remasking}"