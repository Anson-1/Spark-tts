#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer
from pretrained_models.LLADA_8B_Instruct.configuration_llada import LLaDAConfig
from pretrained_models.LLADA_8B_Instruct.modeling_llada import LLaDAModelLM

# Paths (adjust if needed)
tokenizer_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts-llada/tokenizer"
model_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/LLaDA-SMALL"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    print("Loading model config and weights...")
    cfg = LLaDAConfig.from_pretrained(model_path)
    model = LLaDAModelLM.from_pretrained(model_path, config=cfg, trust_remote_code=True)
    model.to(device).eval()

    # Your input (exact string provided)
    input_text = """
<|task_tts|><|start_content|>退啊。<|end_content|><|start_global_token|><|bicodec_global_3339|><|bicodec_global_3574|><|bicodec_global_3656|><|bicodec_global_1714|><|bicodec_global_1624|><|bicodec_global_2889|><|bicodec_global_3622|><|bicodec_global_3117|><|bicodec_global_567|><|bicodec_global_2820|><|bicodec_global_2765|><|bicodec_global_4017|><|bicodec_global_1992|><|bicodec_global_2816|><|bicodec_global_3250|><|bicodec_global_363|><|bicodec_global_3597|><|bicodec_global_842|><|bicodec_global_1308|><|bicodec_global_2060|><|bicodec_global_3485|><|bicodec_global_1800|><|bicodec_global_2688|><|bicodec_global_3051|><|bicodec_global_1784|><|bicodec_global_2926|><|bicodec_global_3982|><|bicodec_global_513|><|bicodec_global_797|><|bicodec_global_1934|><|bicodec_global_3033|><|bicodec_global_2351|><|bicodec_global_4653|><|bicodec_global_4691|><|bicodec_global_1711|><|bicodec_global_3864|><|bicodec_global_5277|><|bicodec_global_6825|><|bicodec_global_1046|><|bicodec_global_844|><|bicodec_global_6077|><|bicodec_global_1571|><|bicodec_global_5661|><|bicodec_global_474|><|bicodec_global_4278|><|bicodec_global_1050|><|bicodec_global_7371|><|bicodec_global_4284|><|bicodec_global_4057|><|bicodec_global_6272|><|bicodec_global_7923|><|bicodec_global_7679|><|bicodec_global_5138|><|bicodec_global_6176|><|bicodec_global_6378|><|bicodec_global_7763|><|bicodec_global_6133|><|bicodec_global_4623|><|bicodec_global_5270|><|bicodec_global_65|><|bicodec_global_7251|><|bicodec_global_4124|><|bicodec_global_5197|><|bicodec_global_3453|><|end_global_token|>
"""

    # Tokenize: keep existing special tokens in text
    try:
        ids = tokenizer.encode(input_text, add_special_tokens=False)
    except Exception:
        # fallback to tokenizer(...) API
        ids = tokenizer(input_text, add_special_tokens=False)['input_ids']

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    print("Input length:", input_ids.shape[1])
    try:
        print("Decoded input (from tokenizer.decode):")
        print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
    except Exception:
        pass

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    print("Logits shape:", logits.shape)

    # Show top-k for the last position
    k = 20
    last_logits = logits[:, -1, :]
    topk = torch.topk(last_logits, k=k, dim=-1)
    topk_ids = topk.indices[0].tolist()
    topk_vals = topk.values[0].tolist()

    print(f"Top-{k} token ids for next position:")
    print(topk_ids)
    try:
        topk_tokens = tokenizer.convert_ids_to_tokens(topk_ids)
        print("Top tokens:")
        print(topk_tokens)
    except Exception:
        print("Could not convert ids to tokens")

    # Also print argmax (greedy next token)
    greedy_id = int(torch.argmax(last_logits, dim=-1).item())
    print("Greedy next token id:", greedy_id)
    try:
        print("Greedy next token string:", tokenizer.convert_ids_to_tokens([greedy_id]))
    except Exception:
        pass

if __name__ == '__main__':
    main()
