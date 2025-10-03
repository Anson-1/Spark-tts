#!/usr/bin/env python3

import sys
sys.path.append('/mnt/lsk_nas/anson/Spark/SparkVox')

from datasets import load_from_disk
from transformers import AutoTokenizer
import torch

# Load the dataset
dataset_path = "/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data/m3ed"
tokenizer_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts/tokenizer"

dataset = load_from_disk(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Check a few examples from train_tts
train_data = dataset['train_tts']
print(f"Total training samples: {len(train_data)}")

# Look at first few examples
for i in range(3):
    example = train_data[i]
    print(f"\n=== Example {i+1} ===")
    print(f"Index: {example['index']}")
    
    # Decode the input_ids and labels to see the actual training data structure
    input_ids = example['input_ids']
    labels = example['labels']
    
    print(f"Input length: {len(input_ids)}")
    print(f"Labels length: {len(labels)}")
    
    # Find where labels transition from -100 to actual values
    first_non_masked = -1
    for j, label in enumerate(labels):
        if label != -100:
            first_non_masked = j
            break
    
    print(f"First non-masked label at position: {first_non_masked}")
    
    # Decode the full sequence to see the structure
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(f"Decoded input: {decoded_input}")
    
    # Decode only the parts that are not masked in labels
    if first_non_masked >= 0:
        target_tokens = [labels[j] if labels[j] != -100 else tokenizer.pad_token_id for j in range(len(labels))]
        decoded_target = tokenizer.decode(target_tokens, skip_special_tokens=False)
        print(f"Decoded target: {decoded_target}")
        
        # Show what the model is expected to generate (non-masked part)
        non_masked_labels = [label for label in labels if label != -100]
        if non_masked_labels:
            decoded_generation_target = tokenizer.decode(non_masked_labels, skip_special_tokens=False)
            print(f"What model should generate: {decoded_generation_target}")
    
    print("-" * 80)

# Check specific tokens
print("\n=== Token Analysis ===")
start_semantic_token = "<|start_semantic_token|>"
im_end_token = "<|im_end|>"
end_semantic_token = "<|end_semantic_token|>"

try:
    start_semantic_id = tokenizer.convert_tokens_to_ids(start_semantic_token)
    im_end_id = tokenizer.convert_tokens_to_ids(im_end_token)
    end_semantic_id = tokenizer.convert_tokens_to_ids(end_semantic_token)
    
    print(f"'{start_semantic_token}' -> {start_semantic_id}")
    print(f"'{im_end_token}' -> {im_end_id}")
    print(f"'{end_semantic_token}' -> {end_semantic_id}")
    
except Exception as e:
    print(f"Error getting token IDs: {e}")