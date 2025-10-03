#!/usr/bin/env python3

import sys
sys.path.append('/mnt/lsk_nas/anson/Spark/SparkVox')

from datasets import load_from_disk
from transformers import AutoTokenizer
import torch

def split_training_data():
    """
    Split the concatenated training data into input/target pairs
    """
    dataset_path = "/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data_dllm/m3ed"
    tokenizer_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts/tokenizer"
    
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Get special token IDs
    start_semantic_id = tokenizer.convert_tokens_to_ids("<|start_semantic_token|>")
    end_semantic_id = tokenizer.convert_tokens_to_ids("<|end_semantic_token|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    print(f"Special token IDs:")
    print(f"  <|start_semantic_token|>: {start_semantic_id}")
    print(f"  <|end_semantic_token|>: {end_semantic_id}")  
    print(f"  <|im_end|>: {im_end_id}")
    
    # Check first example
    train_data = dataset['train_tts']
    example = train_data[0]
    
    input_ids = example['input_ids']
    labels = example['labels']
    
    # Find the split point - look in labels instead of input_ids
    try:
        # The start_semantic_token is the first non-masked token in labels
        start_semantic_idx = -1
        for i, label in enumerate(labels):
            if label == start_semantic_id:
                start_semantic_idx = i
                break
        
        if start_semantic_idx == -1:
            print("Could not find <|start_semantic_token|> in labels")
            return
            
        print(f"\nFound <|start_semantic_token|> at index: {start_semantic_idx}")
        
        # Split into input and target
        input_part = input_ids[:start_semantic_idx + 1]  # Input up to and including where semantic starts
        target_part = labels[start_semantic_idx:]         # Semantic tokens starting from start token
        
        # Remove padding from target
        target_part = [token for token in target_part if token != -100]
        
        # Decode both parts
        input_text = tokenizer.decode(input_part, skip_special_tokens=False)
        target_text = tokenizer.decode(target_part, skip_special_tokens=False)
        
        print(f"\n=== SPLIT RESULT ===")
        print(f"INPUT (text + global + start_token):")
        print(f"Length: {len(input_part)} tokens")
        print(f"Text: {input_text}")
        
        print(f"\nTARGET (semantic tokens only):")
        print(f"Length: {len(target_part)} tokens")
        print(f"Text: {target_text[:200]}..." if len(target_text) > 200 else target_text)
        
        # Extract just the semantic token numbers
        semantic_numbers = []
        for token_id in target_part:
            if token_id != im_end_id:  # Skip <|im_end|>
                token_str = tokenizer.decode([token_id])
                if '<|bicodec_semantic_' in token_str:
                    # Extract number from <|bicodec_semantic_1234|>
                    import re
                    match = re.search(r'<\|bicodec_semantic_(\d+)\|>', token_str)
                    if match:
                        semantic_numbers.append(int(match.group(1)))
        
        print(f"\nExtracted semantic token numbers: {semantic_numbers[:20]}...")
        print(f"Total semantic tokens: {len(semantic_numbers)}")
        
    except ValueError as e:
        print(f"Error: Could not find <|start_semantic_token|> in input_ids: {e}")

if __name__ == "__main__":
    split_training_data()