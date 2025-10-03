#!/usr/bin/env python3

"""
Script to check the training data format and print samples
"""

import datasets
from transformers import AutoTokenizer
import torch

def main():
    # Load the training data
    print("Loading training data...")
    dataset_path = "local/sparktts/train_data_llada/m3ed"
    dataset = datasets.load_from_disk(dataset_path)
    
    # Load tokenizer
    tokenizer_path = "pretrained_models/spark-tts-llada/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"Dataset structure: {dataset}")
    print()
    
    # Check TTS training data
    if 'train_tts' in dataset:
        print("=== TTS Training Data Sample ===")
        tts_data = dataset['train_tts']
        sample = tts_data[0]  # Get first sample
        
        print(f"Sample index: {sample['index']}")
        print(f"Input IDs length: {len(sample['input_ids'])}")
        print(f"Labels length: {len(sample['labels'])}")
        print()
        
        # Decode input
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print("Input text:")
        print(repr(input_text))
        print()
        
        # Decode labels (expected output)
        labels_text = tokenizer.decode(sample['labels'], skip_special_tokens=False)
        print("Expected output (labels):")
        print(repr(labels_text))
        print()
        
        # Check if EOS token is present
        eos_token_id = tokenizer.eos_token_id
        print(f"EOS token: {tokenizer.eos_token} (ID: {eos_token_id})")
        
        if eos_token_id in sample['labels']:
            eos_position = sample['labels'].index(eos_token_id)
            print(f"✓ EOS token found at position {eos_position} in labels")
        else:
            print("❌ EOS token NOT found in labels")
        
        # Check for old im_end token
        im_end_text = "<|im_end|>"
        if im_end_text in labels_text:
            print(f"❌ Old token '<|im_end|>' still present!")
        else:
            print(f"✓ Old token '<|im_end|>' not found")
        
        print()
        print("=== Full sequence format ===")
        full_sequence = sample['input_ids'] + sample['labels']
        full_text = tokenizer.decode(full_sequence, skip_special_tokens=False)
        print("Full training sequence:")
        print(repr(full_text))
        print()
        print(f"Total sequence length: {len(full_sequence)} tokens")
        
    else:
        print("No 'train_tts' data found in dataset")

if __name__ == "__main__":
    main()
