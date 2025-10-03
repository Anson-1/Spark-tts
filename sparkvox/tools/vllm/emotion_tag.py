import argparse
import os
import re
import json
from transformers import AutoTokenizer
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from tqdm import tqdm
import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with Qwen model and produce output JSONL."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/aifs4su/xinshengwang/model/Qwen2.5-72B-Instruct",
        help="Name of the model to load.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="search_results.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to the output JSONL file. If not provided, a default will be used.",
    )
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size.")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of samples to process at once before saving a checkpoint.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file."
    )
    return parser.parse_args()


def is_chinese(text):
    return bool(re.findall(r"[\u4e00-\u9fff]", text))


def read_jsonl(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.

    Args:
    file_path : Path
        The path to the JSONL file to be read.

    Returns:
    List[dict]
        A list of dictionaries parsed from each line of the JSONL file.
    """
    metadata = []
    # Open the file for reading
    with open(file_path, "r", encoding="utf-8") as f:
        # Split the file into lines
        lines = f.read().splitlines()
    # Process each line
    for line in lines:
        # Convert JSON string back to dictionary and append to list
        meta = json.loads(line)
        metadata.append(meta)
    # Return the list of metadata
    return metadata


def build_one_prompt(meta, tokenizer):
    prompt_zh = "请判断以下文本的情感，并选择最合适的标签：[害怕, 开心, 厌恶, 悲伤, 惊讶, 愤怒, 中性]（请注意，只需给出标签，不要提供额外的描述或理由）。文本如下："
    prompt_en = "Please assess the emotion of the following text and select the most appropriate label from these options: [Fearful, Happy, Disgusted, Sad, Surprised, Angry, Neutral]. Please note, only provide the label without any additional description or reasoning. Here is the text:"
    text = meta["text"]
    if is_chinese(text):
        prompt = prompt_zh
    else:
        prompt = prompt_en

    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": f"{prompt}{text}"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text


def build_prompt(data, tokenizer):
    prompts = []
    for d in data:
        prompts.append(build_one_prompt(d, tokenizer))
    return prompts


def main():
    args = parse_args()

    # Initialize Accelerator
    # accelerator = Accelerator()

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    llm = LLM(
        model=args.model_name, tensor_parallel_size=args.tp
    )  # Initialize without tensor_parallel_size here
    # model, tokenizer = accelerator.prepare(
    #     llm, tokenizer
    # )  # Prepare model and tokenizer for multi-GPU

    data = read_jsonl(args.input)

    if os.path.exists(args.output) and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. Use --overwrite to overwrite it."
        )

    sampling_params = SamplingParams(
        temperature=args.temp, top_p=args.top_p, max_tokens=args.max_tokens
    )


    total_data_len = len(data)

    print(f"总数据量: {total_data_len}")
    for i in tqdm(range(0, len(data), args.chunk_size), desc="Processing chunks"):
        chunk = data[i:i + args.chunk_size]
        prompts = [build_one_prompt(d, tokenizer) for d in chunk]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        with open(args.output, "a", encoding="utf-8") as f:
            for d, output in zip(chunk, outputs):
                d['emo_qwen'] = output.outputs[0].text
                # print(d)
                f.write(json.dumps(d, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
