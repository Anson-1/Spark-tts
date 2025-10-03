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

from omegaconf import DictConfig
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
    """
    A data module class compatible with PyTorch Lightning that manages data loading for training, validation, and testing.

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

    def get_loader(self, mode: str, batch_size: int) -> DataLoader:
        """
        Create a data loader for the specified mode.

        Args:
            mode (str): The mode of the dataset to load ('train', 'val', or 'test').
            batch_size (int): batch size

        Returns:
            DataLoader: Configured data loader for the specified dataset mode.
        """
        shuffle = mode == "train"
        drop_last = mode != "train"
        num_workers = self.config["dataloader"]["num_workers"]
        dataset = hydra.utils.instantiate(self.config["dataloader"], self.config, mode)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            persistent_workers=True,
            drop_last=drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self.get_loader("train", self.config["dataloader"]["batch_size"])

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self.get_loader("val", self.config["dataloader"]["val_batch_size"])

    def test_dataloader(self) -> DataLoader:
        """Return the testing DataLoader."""
        return self.get_loader("test", self.config["dataloader"]["val_batch_size"])
