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
# CUDA_VISIBLE_DEVICES=1,2,3,4,5 bash train.sh --config egs/codec/bicodec/config/bicodec.24k.yaml \
#               --log_dir egs/codec/bicodec/results/bicodec.24k \
#               --nproc_per_node 5

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")

# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

# Set default parameters
config="egs/codec/bicodec/config/bicodec.24k.yaml"
log_dir="${script_dir}/results/bicodec.24k"
nnodes=1
nproc_per_node=-1
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
    --version ${version} \\
    --date ${tag}
EOF


chmod +x "$log_dir/${tag}_train.sh"
echo "run bash is saved to $log_dir/${tag}_train.sh"

echo "execute $log_dir/${tag}_train.sh"

bash "$log_dir/${tag}_train.sh"