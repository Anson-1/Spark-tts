# SparkTTS

## Overview

**SparkTTS** is a text-to-speech (TTS) system based on [Bicodec](../../../egs/codec/bicodec/README.md) and Qwen2.5. This README provides a step-by-step guide for data preparation, model setup, and training.

---

## Data Preparation

The metadata should be organized in a `.jsonl` (JSON Lines) format, where each line is a JSON object. Each object must contain at least the following keys:

- `index`: a unique identifier for the sample
- `wav_path`: the path to the waveform file

An example metadata file can be found at:  
`egs/data/metadata/m3ed.jsonl`

---

## Download pre-trained models

- Infer to the tutorial of [Bicodec](../../../egs/codec/bicodec/README.md) for the prepare of audio tokenizer.
- Download the [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/) to `pretrained_models/Qwen2.5-0.5B-Instruct`

## Data prepare

- Expand base tokenizer: `bash egs/speech_synthesis/spark-tts/00.expand_tokenizer.sh`
- Extract audio tokens: `bash egs/speech_synthesis/spark-tts/01.extract_tokens.sh`
- Prepare train data: `bash egs/speech_synthesis/spark-tts/02.prepare_train_data.sh`

## Training

To start training SparkTTS, run the following commands:

```bash
bash egs/speech_synthesis/spark-tts/03.train.sh
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
bash train.sh --egs/speech_synthesis/spark-tts/config/spark-tts_qwen0.5b.yaml
```

## Inference

The script `egs/speech_synthesis/spark-tts/04.infer.sh` provides an example of how to use the trained model for inference. The audio tokenizer should be defined in `egs/speech_synthesis/spark-tts/config/audio_tokenizer/bicodec.yaml`.
