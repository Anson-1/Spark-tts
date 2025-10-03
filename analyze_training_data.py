#!/usr/bin/env python3
"""
Script to analyze LLaDA training data for stop token patterns and repetition issues.
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

def load_tokenizer(tokenizer_path: str):
    """Load the tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def analyze_token_patterns(token_ids: List[int], tokenizer, max_examples: int = 5) -> Dict[str, Any]:
    """Analyze patterns in a sequence of token IDs"""
    
    # Decode tokens for analysis
    try:
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    except:
        decoded = "DECODE_ERROR"
    
    analysis = {
        "length": len(token_ids),
        "unique_tokens": len(set(token_ids)),
        "repetition_ratio": 1.0 - (len(set(token_ids)) / len(token_ids)) if token_ids else 0,
        "decoded_sample": decoded[:200] + "..." if len(decoded) > 200 else decoded,
        "token_counts": Counter(token_ids),
        "has_endoftext": 126081 in token_ids,  # <|endoftext|> token ID
        "has_startoftext": 126080 in token_ids,  # <|startoftext|> token ID
        "bicodec_tokens": 0,
        "semantic_tokens": 0,
        "global_tokens": 0,
        "repetitive_sequences": [],
    }
    
    # Count bicodec tokens
    for token_id in token_ids:
        try:
            token_str = tokenizer.decode([token_id])
            if "bicodec_semantic" in token_str:
                analysis["semantic_tokens"] += 1
            elif "bicodec_global" in token_str:
                analysis["global_tokens"] += 1
            elif "bicodec" in token_str:
                analysis["bicodec_tokens"] += 1
        except:
            pass
    
    # Find repetitive sequences (same token repeated 3+ times)
    current_token = None
    current_count = 0
    for token in token_ids:
        if token == current_token:
            current_count += 1
        else:
            if current_count >= 3:
                try:
                    token_str = tokenizer.decode([current_token])
                    analysis["repetitive_sequences"].append({
                        "token_id": current_token,
                        "token_str": token_str,
                        "count": current_count
                    })
                except:
                    analysis["repetitive_sequences"].append({
                        "token_id": current_token,
                        "token_str": "DECODE_ERROR",
                        "count": current_count
                    })
            current_token = token
            current_count = 1
    
    # Check the last sequence
    if current_count >= 3:
        try:
            token_str = tokenizer.decode([current_token])
            analysis["repetitive_sequences"].append({
                "token_id": current_token,
                "token_str": token_str,
                "count": current_count
            })
        except:
            analysis["repetitive_sequences"].append({
                "token_id": current_token,
                "token_str": "DECODE_ERROR",
                "count": current_count
            })
    
    return analysis

def analyze_training_data(data_path: str, tokenizer_path: str, max_samples: int = 1000):
    """Analyze the training data for patterns"""
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return
    
    print(f"Loading dataset from: {data_path}")
    try:
        dataset = load_from_disk(data_path)
        print(f"Dataset loaded. Keys: {dataset.keys() if hasattr(dataset, 'keys') else 'Single dataset'}")
        
        # Handle different dataset structures
        if hasattr(dataset, 'keys'):
            # If it's a DatasetDict, analyze each split
            for split_name, split_data in dataset.items():
                print(f"\n=== Analyzing {split_name} split ===")
                analyze_split(split_data, tokenizer, max_samples, split_name)
        else:
            # Single dataset
            print(f"\n=== Analyzing dataset ===")
            analyze_split(dataset, tokenizer, max_samples, "dataset")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

def analyze_split(dataset, tokenizer, max_samples: int, split_name: str):
    """Analyze a single dataset split"""
    
    print(f"Split size: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    
    # Overall statistics
    total_samples = min(len(dataset), max_samples)
    endoftext_count = 0
    startoftext_count = 0
    repetitive_samples = 0
    total_length = 0
    total_unique_ratio = 0
    
    repetition_examples = []
    no_endoftext_examples = []
    good_examples = []
    
    print(f"Analyzing {total_samples} samples...")
    
    for i in range(total_samples):
        try:
            sample = dataset[i]
            
            # Get token IDs (check both 'input_ids' and 'labels')
            token_ids = None
            if 'labels' in sample and sample['labels'] is not None:
                token_ids = sample['labels']
            elif 'input_ids' in sample and sample['input_ids'] is not None:
                token_ids = sample['input_ids']
            
            if token_ids is None:
                continue
                
            # Convert to list if it's a tensor
            if hasattr(token_ids, 'tolist'):
                token_ids = token_ids.tolist()
            
            # Filter out padding tokens (-100)
            token_ids = [t for t in token_ids if t != -100]
            
            if not token_ids:
                continue
            
            # Analyze this sample
            analysis = analyze_token_patterns(token_ids, tokenizer)
            
            # Update statistics
            total_length += analysis["length"]
            total_unique_ratio += (1.0 - analysis["repetition_ratio"])
            
            if analysis["has_endoftext"]:
                endoftext_count += 1
            else:
                if len(no_endoftext_examples) < 3:
                    no_endoftext_examples.append({
                        "index": i,
                        "sample": sample.get('index', f'sample_{i}'),
                        "analysis": analysis
                    })
            
            if analysis["has_startoftext"]:
                startoftext_count += 1
            
            if analysis["repetitive_sequences"]:
                repetitive_samples += 1
                if len(repetition_examples) < 3:
                    repetition_examples.append({
                        "index": i,
                        "sample": sample.get('index', f'sample_{i}'),
                        "analysis": analysis
                    })
            
            # Collect good examples (has endoftext, no repetition)
            if analysis["has_endoftext"] and not analysis["repetitive_sequences"]:
                if len(good_examples) < 3:
                    good_examples.append({
                        "index": i,
                        "sample": sample.get('index', f'sample_{i}'),
                        "analysis": analysis
                    })
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total_samples} samples...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # Print summary
    print(f"\n=== {split_name.upper()} ANALYSIS SUMMARY ===")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Average sequence length: {total_length / total_samples:.1f}")
    print(f"Average uniqueness ratio: {total_unique_ratio / total_samples:.3f}")
    print(f"Samples with <|endoftext|>: {endoftext_count}/{total_samples} ({100*endoftext_count/total_samples:.1f}%)")
    print(f"Samples with <|startoftext|>: {startoftext_count}/{total_samples} ({100*startoftext_count/total_samples:.1f}%)")
    print(f"Samples with repetitive sequences: {repetitive_samples}/{total_samples} ({100*repetitive_samples/total_samples:.1f}%)")
    
    # Show examples
    print(f"\n=== GOOD EXAMPLES (with proper endings) ===")
    for i, example in enumerate(good_examples):
        print(f"\nGood Example {i+1}:")
        print(f"  Sample: {example['sample']}")
        print(f"  Length: {example['analysis']['length']}")
        print(f"  Decoded: {example['analysis']['decoded_sample']}")
    
    print(f"\n=== PROBLEMATIC EXAMPLES (no endoftext) ===")
    for i, example in enumerate(no_endoftext_examples):
        print(f"\nNo EndOfText Example {i+1}:")
        print(f"  Sample: {example['sample']}")
        print(f"  Length: {example['analysis']['length']}")
        print(f"  Decoded: {example['analysis']['decoded_sample']}")
    
    print(f"\n=== REPETITIVE EXAMPLES ===")
    for i, example in enumerate(repetition_examples):
        print(f"\nRepetitive Example {i+1}:")
        print(f"  Sample: {example['sample']}")
        print(f"  Length: {example['analysis']['length']}")
        print(f"  Repetitive sequences: {example['analysis']['repetitive_sequences']}")
        print(f"  Decoded: {example['analysis']['decoded_sample']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze LLaDA training data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum samples to analyze")
    
    args = parser.parse_args()
    
    analyze_training_data(args.data_path, args.tokenizer_path, args.max_samples)

if __name__ == "__main__":
    main()
