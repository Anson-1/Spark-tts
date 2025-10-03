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
orgainze text and audio tokens to training format
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
    """
    Read the test indexes from a 'test.txt' file in the same directory as the jsonl file.

    Parameters:
        jsonl_file (Path): Path to the input JSONL file.

    Returns:
        list: A list of test indexes.
    """
    data_dir = Path(jsonl_file).parent
    with open(f"{data_dir}/test.txt", "r") as f:
        lines = f.read().splitlines()
    test_indexes = [line.strip() for line in lines]

    return test_indexes


def process(meta: Dict[str, Any], tokenizer: AutoTokenizer, code_dir: Path, gt_num: int) -> Dict[str, Any]:
    """
    Process each metadata entry into tokenized input and output pairs for training.

    Parameters:
        meta (dict): Metadata dictionary containing 'index' and 'text'.
        tokenizer (AutoTokenizer): The tokenizer for text tokenization.
        code_dir (Path): Directory to load additional token file.
        gt_num (int): Number of global tokens.

    Returns:
        tuple: A tuple containing the input ids, output ids, and index of the processed item.
    """

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

    tts_inputs, tts_ouputs = None, None
    control_tts_inputs, control_tts_outputs = None, None

        
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
        "<|endoftext|>",
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

    
    tts_ouputs = ["<|start_semantic_token|>", semantic_tokens, "<|endoftext|>"]

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
    """
    Process metadata in parallel using multi-threading.

    Parameters:
        metadata (list): A list of metadata entries to process.
        tokenizer (AutoTokenizer): The tokenizer for text tokenization.
        max_workers (int): The number of threads to use.

    Returns:
        list: A list of processed data in the form (index, input_ids, output_ids).
    """

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


def main(
    jsonl_file: Path, save_dir: Path, tokenizer_path: str, code_dir: Path, gt_num: int
):
    """
    Main function to load metadata, process data, and save the dataset.

    Parameters:
        jsonl_file (Path): The path to the input JSONL file.
        save_dir (Path): Directory to save the processed dataset.
        tokenizer_path (str): Path to the tokenizer model.
        code_dir (Path): Directory to load additional token files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    metadata = read_jsonl(jsonl_file)
    test_indexes = [meta["index"] for meta in metadata if meta["split"] == "test"]
    metadata = update_metadata(metadata)
    

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "left"

    tts_data, control_tts_data = process_data_multithread(metadata, tokenizer, code_dir, gt_num)

    splits = defaultdict(list)
    
    for index, input_ids, output_ids in tts_data:
        split = "train_tts" if index not in test_indexes else "val_tts"
        splits[split].append(
            {"index": index, "input_ids": input_ids, "labels": output_ids}
        )

    for index, input_ids, output_ids  in control_tts_data:
        split = "train_control_tts" if index not in test_indexes else "val_control_tts"
        splits[split].append(
            {"index": index, "input_ids": input_ids, "labels": output_ids}
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
    parser = argparse.ArgumentParser(
        description="Process data for task-based training."
    )

    parser.add_argument(
        "--jsonl_file", type=Path, required=True, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the pretrained tokenizer.",
    )
    parser.add_argument(
        "--code_dir",
        type=Path,
        required=True,
        help="Directory where speech token files are stored.",
    )
    parser.add_argument(
        "--gt_num",
        type=int,
        required=True,
        help="Number of global tokens.",
    )

    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(
        jsonl_file=args.jsonl_file,
        save_dir=args.save_dir,
        tokenizer_path=args.tokenizer_path,
        code_dir=args.code_dir,
        gt_num=args.gt_num
    )