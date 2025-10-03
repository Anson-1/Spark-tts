
import torch
import argparse
import os

def inspect_pt_file(file_path, gt_num):
    """
    Loads a .pt file, prints its contents and shape, and shows how it would be split.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting file: {file_path} ---")

    try:
        # Load the tensor from the file
        tokens = torch.load(file_path)
        
        # Check if it's a tensor
        if not isinstance(tokens, torch.Tensor):
            print(f"Error: The file did not contain a PyTorch tensor. Found type: {type(tokens)}")
            return

        total_length = len(tokens)
        print(f"Loaded tensor with shape: {tokens.shape}")
        print(f"Total number of tokens: {total_length}")

        if gt_num is not None:
            print(f"\nSplitting with gt_num (number of global tokens) = {gt_num}")
            if total_length < gt_num:
                print(f"WARNING: Total tokens ({total_length}) is less than gt_num ({gt_num}).")
            
            global_tokens = tokens[:gt_num]
            semantic_tokens = tokens[gt_num:]
            
            num_global = len(global_tokens)
            num_semantic = len(semantic_tokens)
            
            print(f"Number of global tokens found: {num_global}")
            print(f"Number of semantic tokens found: {num_semantic}")

            if num_semantic == 0:
                print("\n>>> RESULT: This file would produce EMPTY semantic tokens. <<<")
            else:
                print("\nThis file appears to contain semantic tokens.")

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a .pt token file.")
    parser.add_argument("file_path", type=str, help="The absolute path to the .pt file to inspect.")
    parser.add_argument("--gt_num", type=int, default=None, help="The number of global tokens (gt_num) to simulate the split.")

    args = parser.parse_args()

    inspect_pt_file(args.file_path, args.gt_num)
