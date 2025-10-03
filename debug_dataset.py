#!/usr/bin/env python3

import sys
sys.path.append('/mnt/lsk_nas/anson/Spark/SparkVox')

from datasets import load_from_disk
from omegaconf import DictConfig

# Check what datasets exist
dataset_path = "/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data/m3ed"
dataset = load_from_disk(dataset_path)

print("Available dataset splits:")
for split in dataset.keys():
    print(f"  {split}: {len(dataset[split])} samples")

# Check which tasks are configured
config_tasks = ['tts', 'control_tts']
print(f"\nConfigured tasks: {config_tasks}")

# Check what would be loaded for training
print(f"\nChecking training data:")
all_datasets = []
for task in config_tasks:
    split_data = f'train_{task}' if len(task) > 0 else 'train'
    print(f"Looking for split: {split_data}")
    if split_data in dataset:
        data_size = len(dataset[split_data])
        print(f"  Found {split_data}: {data_size} samples")
        if data_size > 0:
            all_datasets.append(dataset[split_data])
        else:
            print(f"  WARNING: {split_data} is empty!")
    else:
        print(f"  ERROR: {split_data} not found!")

print(f"\nTotal datasets to concatenate: {len(all_datasets)}")
if len(all_datasets) == 0:
    print("ERROR: No valid datasets found!")
else:
    print("Datasets look OK")