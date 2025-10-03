#!/usr/bin/env python3
"""
Run a quick greedy generation for one example from your TTS dataset.
Save as run_example_generation.py and run from repo root:
    python run_example_generation.py
"""

from datasets import load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch
import sys
import os

# Config - adjust paths if needed
dataset_dir = "/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data_llada/m3ed"
split_name = "train_control_tts"     # or "train_tts"
tokenizer_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts-llada/tokenizer"
model_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/LLaDA-SMALL"
example_index = 0  # which example to inspect

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer(path):
    try:
        print("Loading tokenizer with AutoTokenizer.from_pretrained(...)")
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print("AutoTokenizer failed:", e)
        # fallback to tokenizer.json if present
        tokfile = os.path.join(path, "tokenizer.json")
        if os.path.exists(tokfile):
            print("Falling back to PreTrainedTokenizerFast(tokenizer_file=tokenizer.json)")
            return PreTrainedTokenizerFast(tokenizer_file=tokfile)
        raise

def main():
    ds = load_from_disk(dataset_dir)
    print("Dataset splits:", list(ds.keys()))
    if split_name not in ds:
        print("Split not found:", split_name)
        sys.exit(1)
    split = ds[split_name]
    print(f"Split {split_name} num rows:", len(split))

    tokenizer = load_tokenizer(tokenizer_path)
    print("Tokenizer loaded. Vocab size (approx):", getattr(tokenizer, "vocab_size", None))

    # load model (uses your local HF wrapper class)
    print("Loading model from", model_path)
    from pretrained_models.LLADA_8B_Instruct.configuration_llada import LLaDAConfig
    from pretrained_models.LLADA_8B_Instruct.modeling_llada import LLaDAModelLM

    cfg = LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(model_path, config=cfg, trust_remote_code=True)
    model.to(device).eval()

    # pick example
    ex = split[example_index]
    prompt_ids = ex["input_ids"]
    gt_labels = ex["labels"]

    print("\n== Example index", example_index, "==")
    print("prompt length:", len(prompt_ids), "labels length:", len(gt_labels))
    print("Decoded prompt (skip_special_tokens=True):")
    try:
        print(tokenizer.decode(prompt_ids, skip_special_tokens=True))
    except Exception:
        print("<decode failed>")

    # Show token strings for prompt start
    print("\nFirst 80 prompt token ids/tokens:")
    toks = tokenizer.convert_ids_to_tokens(prompt_ids)
    for i, (tid, tok) in enumerate(zip(prompt_ids, toks)):
        if i >= 80: break
        print(i, tid, tok)

    # Greedy autoregressive generation: feed prompt, generate len(gt_labels) tokens
    max_gen = len(gt_labels)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for step in range(max_gen):
            outputs = model(generated)            # should return an object with .logits
            logits = outputs.logits               # (1, T, V)
            next_logits = logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1, keepdim=True)  # greedy
            generated = torch.cat([generated, next_id], dim=1)

    pred_ids = generated[0, -max_gen:].tolist()
    print("\nPredicted ids (last %d):" % max_gen)
    print(pred_ids)

    # tokens and decode
    try:
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids)
        print("\nPredicted tokens:")
        print(pred_tokens[:200])
    except Exception:
        print("Could not convert pred ids to tokens")

    try:
        decoded_pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
    except Exception:
        decoded_pred = "<decode failed>"
    print("\nDecoded prediction (skip_special_tokens=True):")
    print(decoded_pred)

    # show ground-truth labels tokens/decode for comparison
    try:
        gt_tokens = tokenizer.convert_ids_to_tokens(gt_labels)
        print("\nGround-truth label tokens (first 200):")
        print(gt_tokens[:200])
    except Exception:
        print("Could not convert gt labels to tokens")

    try:
        gt_decoded = tokenizer.decode(gt_labels, skip_special_tokens=True)
    except Exception:
        gt_decoded = "<decode failed>"
    print("\nGround-truth decoded (skip_special_tokens=True):")
    print(gt_decoded)

if __name__ == "__main__":
    main()