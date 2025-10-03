#!/bin/bash

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

cd "$root_dir" || exit

# Run the Python script with the specified arguments
python -m egs.speech_synthesis.spark-tts.scripts.expand_tokenizer_llada \
  --base_tokenizer_path "pretrained_models/LLADA_8B_Instruct" \
  --save_path "pretrained_models/spark-tts-llada/tokenizer"
