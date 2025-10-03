
import torch
import argparse
import os
from transformers import AutoTokenizer

def verify_tokenizer(pt_file_path, tokenizer_path, gt_num):
    """
    Checks if all required semantic tokens from a .pt file exist in the tokenizer's vocabulary.
    """
    print(f"--- Verifying Tokenizer Vocabulary ---")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Token file: {pt_file_path}")

    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        vocab = tokenizer.get_vocab()
        print(f"Tokenizer vocabulary loaded successfully. Size: {len(vocab)} tokens.")
    except Exception as e:
        print(f"ERROR: Could not load tokenizer. {e}")
        return

    # 2. Load the .pt token file
    try:
        tokens = torch.load(pt_file_path)
        semantic_tokens_tensor = tokens[gt_num:]
        print(f"Loaded {len(semantic_tokens_tensor)} semantic token integers from .pt file.")
    except Exception as e:
        print(f"ERROR: Could not load .pt file. {e}")
        return

    # 3. Check each required token
    missing_tokens = []
    for token_int in semantic_tokens_tensor:
        token_str = f"<|bicodec_semantic_{token_int.item()}|>"
        if token_str not in vocab:
            missing_tokens.append(token_str)
    
    # 4. Report results
    print("\n--- Verification Results ---")
    if not missing_tokens:
        print("SUCCESS! All required semantic tokens from this file exist in the tokenizer.")
    else:
        print(f"ERROR: Found {len(missing_tokens)} missing tokens that are needed by this file but are not in the tokenizer!")
        print("This is the cause of the data preparation error.")
        print("\nFirst 10 missing tokens:")
        for i, token in enumerate(missing_tokens[:10]):
            print(f"  {i+1}. {token}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify tokenizer vocabulary against a token file.")
    parser.add_argument("file_path", type=str, help="The absolute path to the .pt file to inspect.")
    parser.add_argument("tokenizer_path", type=str, help="The absolute path to the tokenizer directory.")
    parser.add_argument("--gt_num", type=int, default=64, help="The number of global tokens.")

    args = parser.parse_args()

    verify_tokenizer(args.file_path, args.tokenizer_path, args.gt_num)
