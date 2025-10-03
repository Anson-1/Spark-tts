# Gender Classification

## Overview
This module provides gender classification capabilities for speech audio.

## Supported Models

- WavLM + Lora
  - Uses WavLM as backbone with LoRA fine-tuning
  - Directly processes raw waveform input
- Mel-spectrogram + ECAPA-TDNN
  - Uses mel-spectrogram features with ECAPA-TDNN architecture
  - Efficient and lightweight model option

## How to use

### Training
```bash
bash train.sh --config egs/speaker_attribute/gender/config/mel_gender_predictor.yaml \
              --log_dir egs/speaker_attribute/gender/results/20241203_gender_mel \  
              --nproc_per_node 4
```

### Key Parameters
- `--config`: Configuration file path
- `--log_dir`: Directory for saving training logs and checkpoints
- `--nproc_per_node`: Number of GPUs to use for training


## Model Configurations
Two example configurations are provided:
- `config/wavlm_gender_predictor.yaml`: For WavLM + LoRA model
- `config/mel_gender_predictor.yaml`: For Mel-spectrogram + ECAPA-TDNN model
