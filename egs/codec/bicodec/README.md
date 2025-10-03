# BiCodec

## Overview

**BiCodec**, proposed in the Spark-TTS framework, is a neural codec designed to quantise speech into a single stream of discrete tokens for speech synthesis. It consists of two key modules:

- **Global Tokeniser**: Extracts global tokens from the Mel spectrogram of input audio.
- **Semantic Tokeniser**: Ueses features from wav2vec 2.0 to derive semantic tokens.

This README provides a step-by-step guide for data preparation, model setup, and training.

---

## Data Preparation

The metadata should be organized in a `.jsonl` (JSON Lines) format, where each line is a JSON object. Each object must contain at least the following keys:

- `index`: a unique identifier for the sample
- `wav_path`: the absolute path to the waveform file

An example metadata file can be found at:  
`egs/data/metadata/m3ed.jsonl`

> **Note**: Ensure that `wav_path` is an *absolute* path. Relative paths are not supported.

---

## Download wav2vec 2.0

1. Create a directory named `pretrained_models` in the root of the SparkVox repository.
2. Download the [`wav2vec2-large-xlsr-53`](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) model.
3. Place the downloaded model in the `pretrained_models` directory.

---

## Training

To start training BiCodec, run the following commands:

```bash
cd egs/codec/bicodec
bash train.sh
```

**Script Arguments**

The train.sh script accepts the following arguments:

- --config: Path to the configuration YAML file (relative to project root)

- --log_dir: Directory to store training logs and checkpoints

- --nnodes: Number of nodes for distributed training

- --nproc_per_node: Number of GPUs per node (-1 to use all available GPUs)

- --resume: Resume training from a specific checkpoint. 0 means resume from the last checkpoint if `version` is set

- --version: If not specified, a new version will be created. If specified, training will resume from the checkpoint corresponding to that version

**Example**
```
bash train.sh --config egs/codec/bicodec/config/bicodec.24k.yaml
```

## Inference

The script `egs/codec/bicodec/infer.sh` provides an example of how to use the trained model for inference. The  following arguments should be defined:

- log_dir: log directory same to the training script
- jsonlpath: jsonl file for inference with the same format as the training metadata
- wav2vec_model: path to the wav2vec 2.0 model
- ckpt: path to the checkpoint that to be used for inference