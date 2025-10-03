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

from pathlib import Path
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class DataCollatorForCausalLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer_path: Path,
        padding_side: str = "left",
        mlm: bool = False,
        return_tensors: str ="pt",
        max_length: int = 2048,
        **kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.padding_side = padding_side
        super().__init__(tokenizer, mlm=mlm, return_tensors=return_tensors, **kwargs)
        self.padding_side = padding_side
        self.max_length = max_length
    def __call__(self, examples):
        batch = []
        new_examples = []
        for example in examples:
            full_input_ids = list(example["input_ids"]) + list(example["labels"])
            if len(full_input_ids) > self.max_length: continue
            new_example = {
                "input_ids": full_input_ids,
            }
            batch.append(new_example)
            new_examples.append(example)
        
        # Call the parent class __call__ to handle padding, etc.
        collated = super().__call__(batch)

        labels = collated["labels"].clone()

        for i, example in enumerate(new_examples):
            original_length = len(example["input_ids"]) + len(example["labels"])
            input_length = len(example["input_ids"])

            if self.padding_side == "left":
                # Left-side padding case
                padding_length = collated["input_ids"].size(1) - original_length
                labels[i, : padding_length + input_length] = -100  # Mask input part
                labels[i, -1] = self.tokenizer.eos_token_id  # EOS
            else:
                labels[i, :input_length] = -100  # Mask input part
                labels[i, original_length:] = -100  # Mask padding
                labels[i, original_length - 1] = self.tokenizer.eos_token_id  # EOS

        collated["labels"] = labels
        return collated


if __name__ == "__main__":
    collator = DataCollatorForCausalLM(
        tokenizer_path="/aifs4su/xinshengwang/code/spark-tts/tokenizer/spark-tts-bicodec-pitch-energy-speech-tokenizer",
        mlm=False,
        return_tensors="pt",
    )

    datasets = load_from_disk(
        "/aifs4su/xinshengwang/data/spark-tts/train_data/LibriSpeech"
    )["val"]

    dataloader = DataLoader(
        datasets,
        batch_size=12,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    )

    for batch in dataloader:
        print("input shape:", batch["input_ids"].shape)
        print("label shape:", batch["labels"].shape)