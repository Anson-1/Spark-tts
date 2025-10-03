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
default_log_dir="${script_dir}/results/bicodec.24k/20250420_014312"
log_dir="${default_log_dir}"
device=0

jsonlpath="/aifs4su/xinshengwang/data/speech/vocoder/test/librispeech/test.jsonl"
wav2vec_model='/aifs4su/xinshengwang/model/wav2vec2-large-xlsr-53'
ckpt='egs/codec/bicodec/results/bicodec.24k/20250420_014312/ckpt/epoch=0010_step=110000.ckpt'
step=110000
save_dir="${log_dir}/infer/${step}"
config="${log_dir}/config.yaml"

# Change directory to the root directory
cd "$root_dir" || exit

# Run inference
python -m sparkvox.models.codec.BiCodec.inference \
    --ckpt "${ckpt}" \
    --config "${config}" \
    --save_dir "${save_dir}" \
    --jsonfile "${jsonlpath}" \
    --device "${device}" \
    --wav2vec_model "${wav2vec_model}"