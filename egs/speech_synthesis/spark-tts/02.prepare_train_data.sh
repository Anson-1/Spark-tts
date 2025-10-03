#!/bin/bash

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

cd "$root_dir" || exit

# Loop through all data names and corresponding JSONL files

data_name='m3ed'
jsonl_file="egs/data/metadata/${data_name}.jsonl"
codec_dir="local/${data_name}"
save_dir="local/sparktts/train_data_dllm/${data_name}"
tokenizer_path="/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts/tokenizer"

echo ${data_name}

# Run the Python script with the specified arguments
python -m egs.speech_synthesis.spark-tts.scripts.prepare_train \
--jsonl_file "$jsonl_file" \
--save_dir "$save_dir" \
--tokenizer_path "$tokenizer_path" \
--code_dir "$codec_dir" \
--gt_num 64