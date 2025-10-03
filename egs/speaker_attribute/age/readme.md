# Age Classification

## Overview
This module provides age classification capabilities for speech audio.

## Age Groups
We classify speakers into the following age groups:
- Child: 0-12
- Teenager: 12-18
- Youth-Adult: 18-40
- Middle-aged: 40-60
- Elderly: 60+

Please refer to `utils/age_parser.py` for more details.

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
bash train.sh --config egs/speaker_attribute/age/config/mel_age_predictor.yaml \
              --log_dir egs/speaker_attribute/age/results/20241203_age_mel \  
              --nproc_per_node 4
```

### Key Parameters
- `--config`: Configuration file path
- `--log_dir`: Directory for saving training logs and checkpoints
- `--nproc_per_node`: Number of GPUs to use for training


## Model Configurations
Two example configurations are provided:
- `config/wavlm_age_predictor.yaml`: For WavLM + LoRA model
- `config/mel_age_predictor.yaml`: For Mel-spectrogram + ECAPA-TDNN model

