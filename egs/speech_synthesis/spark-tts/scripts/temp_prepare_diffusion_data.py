# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
orgainze text and audio tokens to training format for diffusion/masking training.
"""

import re
import os
import torch
import argparse
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, DatasetDict

from sparkvox.utils.file import read_jsonl
from sparkvox.models.speech_synthesis.sparktts.utils.token_parser import (
    TASK_TOKEN_MAP,
    LEVELS_MAP,
    GENDER_MAP,
    AGE_MAP,
    EMO_MAP
)
from sparkvox.utils.attribute_parser import AttributeParser


def is_chinese(text):
    return bool(re.findall(r"[\u4e00-\u9fff]", text))

def update_metadata(metadata: list) -> list:
    new_metadata = []
    for meta in tqdm(metadata, desc="Updating metadata"):
        text = meta.get('text', '')
        speed = meta['syllable_num'] / meta['speech_duration'] 
        if speed > 10: continue
        lang = 'zh' if is_chinese(text) else 'en'
        speed_label, speed_value = AttributeParser.speed_token(speed, lang=lang)
        mel_label, mel_value = AttributeParser.pitch_token(meta['pitch'], meta['gender'])
        pitch_std_label, pitch_std_value = AttributeParser.pitch_std_token(meta['pitch_std'], meta['gender'])
 
    
        age_id = AGE_MAP[meta['age']]
        gender_id = GENDER_MAP[meta['gender']]
        emotion_id = EMO_MAP[meta['emotion'].upper()]
        pitch_level_id = LEVELS_MAP[mel_label]
        pitch_std_level_id = LEVELS_MAP[pitch_std_label]
        speed_level_id = LEVELS_MAP[speed_label]

        attribte_ids = [age_id, gender_id, emotion_id, pitch_level_id, pitch_std_level_id, speed_level_id]
        acoustic_ids = [mel_value, pitch_std_value, speed_value]

        new_meta = {
            'index': meta['index'],
            'language': lang,
            'age': meta['age'],
            'gender': meta['gender'],
            'emotion': meta['emotion'],
            'pitch': meta['pitch'],
            'pitch_std': meta['pitch_std'],
            'speed': np.round(speed, 1),
            'duration': meta['duration'],
            'speech_duration': meta['speech_duration'],
            'syllable_num': meta['syllable_num'],
            'text': text,
            'syllables': meta['syllables'],
            'attribte_ids': attribte_ids,
            'acoustic_ids': acoustic_ids,
            'wav_path': meta['wav_path']
        }

        new_metadata.append(new_meta)
    
    return new_metadata


def get_test_indexes(jsonl_file: Path) -> list:
    data_dir = Path(jsonl_file).parent
    with open(f"{data_dir}/test.txt", "r") as f:
        lines = f.read().splitlines()
    test_indexes = [line.strip() for line in lines]

    return test_indexes


def process(meta: Dict[str, Any], tokenizer: AutoTokenizer, code_dir: Path, gt_num: int) -> Dict[str, Any]:
    tokens = torch.load(f'{code_dir}/{meta["index"]}.pt')
    global_tokens, semantic_tokens = tokens[:gt_num], tokens[gt_num:]
    global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_tokens])
    semantic_tokens = "".join(
        [f"<|bicodec_semantic_{i}|>" for i in semantic_tokens]
    )
    attribte_tokens, acoustic_tokens = [], []

    for tag, id in zip(
        [
            "age",
            "gender",
            "emotion",
            "pitch_label",
            "pitch_var_label",
            "speed_label",
        ],
        meta["attribte_ids"],
    ):
        attribte_tokens.append(f"<|{tag}_{id}|>")
    
    gender_token, pitch_token, speed_token = attribte_tokens[1], attribte_tokens[3], attribte_tokens[5]
    attribte_tokens = [gender_token, pitch_token, speed_token]

    for tag, id in zip(
        ["pitch_value", "pitch_var_value", "speed_value"],
        meta["acoustic_ids"],
    ):
        acoustic_tokens.append(f"<|{tag}_{id}|>")

    pitch_value_token, speed_value_token = acoustic_tokens[0], acoustic_tokens[2]
    acoustic_tokens = [pitch_value_token, speed_value_token]

    attribte_tokens = "".join(attribte_tokens)
    acoustic_tokens = "".join(acoustic_tokens)

    control_tts_inputs = [
        TASK_TOKEN_MAP["controllable_tts"],
        "<|start_content|>",
        meta["text"],
        "<|end_content|>",
        "<|start_style_label|>",
        attribte_tokens,
        "<|end_style_label|>",
    ]

    control_tts_outputs = [
        "<|start_acoustic_token|>",
        acoustic_tokens,
        "<|end_acoustic_token|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        semantic_tokens,
        "<|im_end|>",
    ]

    tts_inputs = [
        TASK_TOKEN_MAP["tts"],
        "<|start_content|>",
        meta["text"],
        "<|end_content|>",
        "<|start_global_token|>",
        global_tokens,
        "<|end_global_token|>",
    ]

    
    tts_ouputs = ["<|start_semantic_token|>", semantic_tokens, "<|im_end|>"]

    tts_inputs = "".join(tts_inputs)
    tts_inputs_id = tokenizer.encode(
        tts_inputs,
        add_special_tokens=False,
        padding=False,
        truncation=True,
    )

    tts_ouputs = "".join(tts_ouputs)
    tts_ouputs_id = tokenizer.encode(
        tts_ouputs,
        add_special_tokens=False,
        padding=False,
        truncation=True,
    )
    
    # Validate token IDs
    vocab_size = tokenizer.vocab_size
    all_tts_ids = tts_inputs_id + tts_ouputs_id
    for token_id in all_tts_ids:
        if token_id >= vocab_size:
            raise ValueError(f'TTS Token ID {token_id} is out of bounds for vocab size {vocab_size} in index {meta["index"]}')

    tts_data = {"input": tts_inputs_id, "output": tts_ouputs_id}

    control_tts_inputs = "".join(control_tts_inputs)
    control_tts_inputs_id = tokenizer.encode(
        control_tts_inputs,
        add_special_tokens=False,
        padding=False,
        truncation=True,
    )

    control_tts_outputs = "".join(control_tts_outputs)
    control_tts_outputs_id = tokenizer.encode(
        control_tts_outputs,
        add_special_tokens=False,
        padding=False,
        truncation=True,
    )

    # Validate token IDs
    all_control_ids = control_tts_inputs_id + control_tts_outputs_id
    for token_id in all_control_ids:
        if token_id >= vocab_size:
            raise ValueError(f'Control TTS Token ID {token_id} is out of bounds for vocab size {vocab_size} in index {meta["index"]}')

    control_tts_data = {
            "input": control_tts_inputs_id,
            "output": control_tts_outputs_id,
        }

    return {
        "index": meta["index"],
        "tts_data": tts_data,
        "control_tts_data": control_tts_data,
    }
   

def process_data_multithread(metadata, tokenizer, code_dir, gt_num, max_workers=128):
    tts_data = []
    control_tts_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process, meta, tokenizer, code_dir, gt_num)
            for meta in metadata
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            outputs = future.result()
            index = outputs['index']
            if outputs['control_tts_data'] is not None:
                control_tts_data.append((index, outputs["control_tts_data"]['input'], outputs["control_tts_data"]['output']))
            if outputs['tts_data'] is not None:
                tts_data.append((index, outputs["tts_data"]["input"], outputs["tts_data"]["output"]))
           
    return tts_data, control_tts_data


from tokenizers import Tokenizer
from transformers import Qwen2TokenizerFast


def main(
    jsonl_file: Path, save_dir: Path, tokenizer_path: str, code_dir: Path, gt_num: int
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    metadata = read_jsonl(jsonl_file)
    test_indexes = [meta["index"] for meta in metadata if meta["split"] == "test"]
    metadata = update_metadata(metadata)
    

    # Load the tokenizer directly from the .json file to bypass caching
    tk_obj = Tokenizer.from_file(os.path.join(tokenizer_path, 'tokenizer.json'))
    tokenizer = Qwen2TokenizerFast(tokenizer_object=tk_obj)

    if tokenizer.vocab_size != 165158:
        raise ValueError(f'Loaded tokenizer has incorrect vocab size! Expected 165158, but got {tokenizer.vocab_size}')
    tokenizer.padding_side = "left"

    tts_data, control_tts_data = process_data_multithread(metadata, tokenizer, code_dir, gt_num)

    splits = defaultdict(list)
    
    # MODIFIED FOR DIFFUSION TRAINING
    # We concatenate inputs and outputs into a single sequence.
    for index, input_ids, output_ids in tts_data:
        split = "train_tts" if index not in test_indexes else "val_tts"
        combined_ids = input_ids + output_ids
        splits[split].append(
            {"index": index, "input_ids": combined_ids, "labels": combined_ids}
        )

    # MODIFIED FOR DIFFUSION TRAINING
    # We concatenate inputs and outputs into a single sequence.
    for index, input_ids, output_ids  in control_tts_data:
        split = "train_control_tts" if index not in test_indexes else "val_control_tts"
        combined_ids = input_ids + output_ids
        splits[split].append(
            {"index": index, "input_ids": combined_ids, "labels": combined_ids}
        )
    
    train_dataset_tts = Dataset.from_list(splits["train_tts"])
    validation_dataset_tts = Dataset.from_list(splits["val_tts"])
    train_dataset_control_tts = Dataset.from_list(splits["train_control_tts"])
    validation_dataset_control_tts = Dataset.from_list(splits["val_control_tts"])
    
    dataset= {}
    
    if train_dataset_tts.num_rows != 0:
        dataset.update({'train_tts': train_dataset_tts})
    
    if validation_dataset_tts.num_rows != 0:
        dataset.update({'validation_tts': validation_dataset_tts})

    if train_dataset_control_tts.num_rows != 0:
        dataset.update({'train_control_tts': train_dataset_control_tts})
    if validation_dataset_control_tts.num_rows != 0:
        dataset.update({'validation_control_tts': validation_dataset_control_tts})

    dataset_dict = DatasetDict(dataset)
    dataset_dict.save_to_disk(save_dir)


if __name__ == "__main__":
    # Hardcoded paths for simplicity - using absolute paths
    jsonl_file_path = Path("/mnt/lsk_nas/anson/Spark/SparkVox/egs/data/metadata/m3ed.jsonl")
    save_dir_path = Path("/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data/m3ed/")
    tokenizer_path_str = "/mnt/lsk_nas/anson/Spark/Open-dLLM/scripts/final_tokenizer"
    code_dir_path = Path("/mnt/lsk_nas/anson/Spark/SparkVox/local/m3ed")
    gt_num_int = 4096

    print("Starting data preparation with hardcoded paths...")
    # Run the main function with the hardcoded paths
    main(
        jsonl_file=jsonl_file_path,
        save_dir=save_dir_path,
        tokenizer_path=tokenizer_path_str,
        code_dir=code_dir_path,
        gt_num=gt_num_int
    )
    print("Data preparation finished.")