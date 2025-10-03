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
#
# Example
# CUDA_VISIBLE_DEVICES=1,2,3,4,5 bash train.sh --config egs/speech_synthesis/spark-tts/config/spark-tts_qwen0.5b.yaml \
#               --log_dir egs/speech_synthesis/spark-tts/results/sparktts_qwen0.5b \
#               --nproc_per_node 5

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export PYTHONWARNINGS="ignore"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

# Set default parameters
config="egs/speech_synthesis/spark-tts/config/spark-tts_qwen0.5b.yaml"
log_dir="local/sparktts/logs"
nnodes=1
nproc_per_node=1
num_workers=8
accumluate=12
resume=0
version=null
port=10086

cd "$root_dir" || exit
source sparkvox/utils/parse_options.sh

# Check if log_dir is already an absolute path
if [[ "$log_dir" != /* ]]; then
    # If log_dir is a relative path, prepend root_dir to it
    log_dir="$root_dir/$log_dir"
fi

# Check if log directory exists
if [ $resume -eq 0 ]; then
    if [ ! -d "$log_dir" ]; then
        mkdir -p "$log_dir"
        echo "Log directory created: $log_dir"
    fi
fi

# Write command to train.bash
tag="$(date +'%Y%m%d_%H%M%S')"

if [ "$version" = "null" ]; then
    version="$tag"
fi

cat <<EOF > "$log_dir/${tag}_train.sh"
#!/bin/bash

# # Change directory to the root directory
cd "$root_dir" || exit

python -m bins.train_pl \\
    --config ${config} \\
    --log_dir ${log_dir} \\
    --resume ${resume} \\
    --nnodes ${nnodes} \\
    --nproc_per_node ${nproc_per_node} \\
    --accumluate ${accumluate} \\
    --version ${version} \\
    --date ${tag}
EOF


chmod +x "$log_dir/${tag}_train.sh"
echo "run bash is saved to $log_dir/${tag}_train.sh"

echo "execute $log_dir/${tag}_train.sh"

bash "$log_dir/${tag}_train.sh"