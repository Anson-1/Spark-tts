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

import hydra
import pytorch_lightning as pl

from typing import List
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets


class MultiDictData(pl.LightningDataModule):
    """
    A data module class compatible with PyTorch Lightning that manages data loading for training and validation.

    Attributes:
        config (DictConfig): Configuration for the dataset.
    """

    def __init__(
        self,
        config: DictConfig,
        **kwargs,
    ) -> None:
        """Initialize the data module with dataset configuration, batch size, and worker details."""
        super().__init__()

        self.config = config
        self.collator = hydra.utils.instantiate(config["collator"])

    def get_loader(self, split: str, batch_size: int, tasks: List[str] = ['']) -> DataLoader:
        """
        Create a data loader for the specified mode.

        Args:
            split (str): The split of the dataset to load ('train', 'val', or 'test').
        """

        shuffle = split == "train"
        drop_last = split != "train"
        num_workers = self.config["dataloader"]["num_workers"]

        all_datasets = []
        
        for task in tasks:
            split_data =  f'{split}_{task}' if len(task) > 0 else split
            for path in self.config["data_paths"][split]:
                dataset = load_from_disk(path)
                if split_data not in dataset: continue
                dataset = dataset[split_data]
                all_datasets.append(dataset)

        all_datasets = concatenate_datasets(all_datasets, axis=0)

        return DataLoader(
            all_datasets,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            num_workers=num_workers,
            persistent_workers=True,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self.get_loader("train", self.config["dataloader"]["batch_size"], self.config["task"])

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self.get_loader("validation", self.config["dataloader"]["val_batch_size"], self.config["task"])


# test
if __name__ == "__main__":
    from tqdm import tqdm
    from sparkvox.utils.file import load_config

    config = load_config(
        "egs/speech_synthesis/spark-tts/config/sparktts_qwen0.5b.yaml"
    )
    dataset = hydra.utils.instantiate(config["datasets"], config["datasets"])

    val_dataset = dataset.train_dataloader()
    # val_dataset = dataset.val_dataloader()
    max_value = 0
    for batch in tqdm(val_dataset):
        if batch['input_ids'].max() > max_value:
            max_value = batch['input_ids'].max()
            print(max_value)
        if batch['labels'].max() > max_value:
            max_value = batch['labels'].max()
            print(max_value)
        # print("batch size", batch["labels"].shape)
    print('max_value', max_value)
