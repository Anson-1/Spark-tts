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
expand existing tokenizer with special tokens and audio tokens.
"""

import argparse

from pathlib import Path
from transformers import AutoTokenizer


def extend_tokenizer(
    base_tokenizer_path: Path,
    save_path: Path,
    num_global_tokens: int,
    num_semantic_tokens: int,
):
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)

    global_tokens = [f"<|bicodec_global_{i}|>" for i in range(num_global_tokens)]
    semantic_tokens = [f"<|bicodec_semantic_{i}|>" for i in range(num_semantic_tokens)]
    pitch_value_tokens = [f"<|pitch_value_{i}|>" for i in range(1001)] + [
        f"<|pitch_var_value_{i}|>" for i in range(11)
    ]
    loudness_value_tokens = [f"<|loudness_value_{i}|>" for i in range(31)] 
    speed_value_tokens = [f"<|speed_value_{i}|>" for i in range(11)] 

    pitch_label_tokens = [f"<|pitch_label_{i}|>" for i in range(5)] + [
        f"<|pitch_var_label_{i}|>" for i in range(5)
    ]
    loudness_label_tokens = [f"<|loudness_label_{i}|>" for i in range(5)] 
    speed_label_tokens = [f"<|speed_label_{i}|>" for i in range(5)] 

    emotion_tokens = [f"<|emotion_{i}|>" for i in range(101)]
    age_tokens = [f"<|age_{i}|>" for i in range(5)] 
    gender_tokens = [f"<|gender_{i}|>" for i in range(3)] 

    special_tokens = [
        "<|task_vc|>",
        "<|task_tts|>",
        "<|task_asr|>",
        "<|task_s2s|>",
        "<|task_t2s|>",
        "<|task_cap|>",
        "<|task_understand|>",
        "<|task_controllable_tts|>",   # label-based
        "<|task_prompt_tts|>",         # nlp description-based
        "<|task_edit|>",
        "<|start_content|>",
        "<|start_style_label|>",
        "<|start_style_prompt|>",
        "<|start_acoustic_token|>",
        "<|start_global_token|>",
        "<|start_semantic_token|>",
        "<|end_content|>",
        "<|end_style_label|>",
        "<|end_style_prompt|>",
        "<|end_acoustic_token|>",
        "<|end_global_token|>",
        "<|end_semantic_token|>",
    ]

    additional_tokens = (
        global_tokens
        + semantic_tokens
        + pitch_value_tokens
        + pitch_label_tokens
        + speed_value_tokens
        + speed_label_tokens
        + loudness_value_tokens
        + loudness_label_tokens
        + age_tokens
        + gender_tokens
        + emotion_tokens
        + special_tokens
    )

    print(f"Original vocab size: {len(tokenizer)}")

    num_added_tokens = tokenizer.add_tokens(additional_tokens)
    print(f"Added {num_added_tokens} tokens to the tokenizer.")

    # save expanded tokenizer
    tokenizer.save_pretrained(save_path)

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update tokenizer.")

    parser.add_argument(
        "--base_tokenizer_path",
        type=Path,
        default="pretrained_models/Qwen2.5-0.5B-Instruct",
        help="Path to the base tokenizer.",
    )

    parser.add_argument(
        "--save_path",
        type=Path,
        default="pretrained_models/spark-tts/tokenizer",
        help="Path where the output will be saved.",
    )

    parser.add_argument(
        "--num_global_tokens", type=int, default=4096, help="Number of global tokens."
    )

    parser.add_argument(
        "--num_semantic_tokens",
        type=int,
        default=8192,
        help="Number of semantic tokens.",
    )

    args = parser.parse_args()

    extend_tokenizer(
        args.base_tokenizer_path,
        args.save_path,
        args.num_global_tokens,
        args.num_semantic_tokens,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.save_path)

    print(f"New vocab size: {len(tokenizer)}")
    inputs = [
        "<|task_prompt_tts|>",
        "<|start_content|>",
        "hello, here is a test for 可控语音合成。",
        "<|end_content|>",
        "<|start_style_label|>",
        "<|gender_0|>",
        "<|age_0|>",
        "<|emotion_0|>",
        "<|pitch_label_0|>",
        "<|pitch_var_label_0|>",
        "<|loudness_label_0|>",
        "<|speed_label_0|>",
        "<|end_style_label|>",
        "<|start_style_prompt|>",
        "一个中年女性的声音，音调较高，情绪高涨。",
        "<|end_style_prompt|>",
        "<|start_acoustic_token|>",
        "<|pitch_value_0|><|pitch_var_value_0|><|loudness_value_0|><|speed_value_0|>",
        "<|end_acoustic_token|>",
        "<|start_global_token|>",
        "<|bicodec_global_10|><|bicodec_global_20|><|bicodec_global_31|>",
        "<|end_global_token|>",
        "<|start_semantic_token|>",
        "<|bicodec_semantic_311|><|bicodec_semantic_240|><|bicodec_semantic_1150|><|bicodec_semantic_8191|>",
        "<|im_end|>",
    ]

    for input in inputs:
        print('input', input)
        ids = tokenizer.encode(input, add_special_tokens=False)
        print(ids)
        print('decode', tokenizer.decode(ids))


## ouput
# Original vocab size: 151665
# Added 13493 tokens to the tokenizer.
# New vocab size: 165158
# input <|task_prompt_tts|>
# [165144]
# decode <|task_prompt_tts|>
# input <|start_content|>
# [165146]
# decode <|start_content|>
# input hello, here is a test for 可控语音合成。
# [14990, 11, 1588, 374, 264, 1273, 369, 26853, 107, 99332, 105761, 106726, 1773]
# decode hello, here is a test for 可控语音合成。
# input <|end_content|>
# [165152]
# decode <|end_content|>
# input <|start_style_label|>
# [165147]
# decode <|start_style_label|>
# input <|gender_0|>
# [165032]
# decode <|gender_0|>
# input <|age_0|>
# [165027]
# decode <|age_0|>
# input <|emotion_0|>
# [165035]
# decode <|emotion_0|>
# input <|pitch_label_0|>
# [164965]
# decode <|pitch_label_0|>
# input <|pitch_var_label_0|>
# [164970]
# decode <|pitch_var_label_0|>
# input <|loudness_label_0|>
# [165022]
# decode <|loudness_label_0|>
# input <|speed_label_0|>
# [164986]
# decode <|speed_label_0|>
# input <|end_style_label|>
# [165153]
# decode <|end_style_label|>
# input <|start_style_prompt|>
# [165148]
# decode <|start_style_prompt|>
# input 一个中年女性的声音，音调较高，情绪高涨。
# [46944, 15946, 7948, 101968, 104668, 3837, 78685, 47872, 105540, 3837, 104405, 114528, 1773]
# decode 一个中年女性的声音，音调较高，情绪高涨。
# input <|end_style_prompt|>
# [165154]
# decode <|end_style_prompt|>
# input <|start_acoustic_token|>
# [165149]
# decode <|start_acoustic_token|>
# input <|pitch_value_0|><|pitch_var_value_0|><|loudness_value_0|><|speed_value_0|>
# [163953, 164954, 164991, 164975]
# decode <|pitch_value_0|><|pitch_var_value_0|><|loudness_value_0|><|speed_value_0|>
# input <|end_acoustic_token|>
# [165155]
# decode <|end_acoustic_token|>
# input <|start_global_token|>
# [165150]
# decode <|start_global_token|>
# input <|bicodec_global_10|><|bicodec_global_20|><|bicodec_global_31|>
# [151675, 151685, 151696]
# decode <|bicodec_global_10|><|bicodec_global_20|><|bicodec_global_31|>
# input <|end_global_token|>
# [165156]
# decode <|end_global_token|>
# input <|start_semantic_token|>
# [165151]
# decode <|start_semantic_token|>
# input <|bicodec_semantic_311|><|bicodec_semantic_240|><|bicodec_semantic_1150|><|bicodec_semantic_8191|>
# [156072, 156001, 156911, 163952]
# decode <|bicodec_semantic_311|><|bicodec_semantic_240|><|bicodec_semantic_1150|><|bicodec_semantic_8191|>
# input <|im_end|>
# [151645]
# decode <|im_end|>