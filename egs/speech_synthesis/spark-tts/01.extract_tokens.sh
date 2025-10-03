#!/bin/bash

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

cd "$root_dir" || exit

# Set the batch size for processing
# batch_size=32
batch_size=4
data_name='m3ed'
jsonlfile="egs/data/metadata/m3ed.jsonl"
save_dir="local/${data_name}"

config_path="/mnt/lsk_nas/anson/Spark/SparkVox/egs/codec/bicodec/config/bicodec.24k.yaml"
ckpt_path="/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/Spark-TTS-0.5B/BiCodec/model.safetensors"
data_root="$root_dir/egs/data/audios"

# Run the Python script with the specified arguments
python -m sparkvox.tools.tokenizer.audio_tokenizer.bicodec.extract_codes \
  --jsonlfile "$jsonlfile" \
  --data_root "$data_root" \
  --config_path "$config_path" \
  --ckpt_path "$ckpt_path" \
  --save_dir "$save_dir" \
  --batch_size "$batch_size" \
  --world_size 1     # Use 1 GPU

