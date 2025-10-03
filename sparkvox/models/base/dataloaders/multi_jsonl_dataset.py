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


import os

from typing import Dict, Any
from pathlib import Path
from typing import List
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset

from sparkvox.utils.file import read_jsonl


class BaseDataset(Dataset):
    """
    Dataset class for managing data in either training or validation modes.

    Attributes:
        config (DictConfig): Configuration containing dataset parameters.
        mode (str): Operating mode of the dataset, 'train' or 'val'.
        metadata (List[dict]): List of metadata dictionaries loaded from jsonl files.
    """

    def __init__(self, config: DictConfig, mode: str = "train", extract_feat: bool = False, data_root: Path = None, **kwargs) -> None:
        """
        Initialize the dataset with specific configuration and mode.

        Args:
            config (DictConfig): Dataset configuration as a dictionary.
            mode (str, optional): Specifies the mode, 'train' or 'val'. Defaults to 'train'.
        """
        self.config = config
        self.mode = mode
        self.data_root = data_root
        self.train = mode == "train" and not extract_feat
        if extract_feat:
            self.config["jsonlfiles"]["val"] = config["jsonlfiles_for_extract"]
        self.metadata = self.load_metadata()

    def get_sample(self, meta: DictConfig) -> Dict[str, Any]:
        """Define this function in the subclass."""
        return {"index": meta["index"]}

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """Return the sample at the given index."""
        return self.get_sample(self.metadata[idx])

    def load_jsonlfiles(self, datalist_file: Path) -> List[str]:
        """
        Load jsonl files from a specified datalist file or single jsonl file.

        Args:
            datalist_file (Path): Path to a datalist file or single jsonl file.

        Returns:
            List[str]: List of jsonl file paths.
        """
        if isinstance(datalist_file, ListConfig):
            return datalist_file

        elif os.path.splitext(datalist_file)[-1] in [".jsonl", ".json", ".collect", ".emotion"]:
            return [datalist_file]
            
        with open(datalist_file, "r") as f:
            return [line.strip() for line in f.readlines()]

    def load_metadata(self) -> List[Dict]:
        """Load metadata from jsonl files.

        Returns:
            List[Dict]: List of metadata extracted from jsonl files.
        """
        if self.mode == "train":
            datalist_file = self.config["jsonlfiles"]["train"]
        else:
            datalist_file = self.config["jsonlfiles"]["val"]
        jsonlfiles = self.load_jsonlfiles(datalist_file)
        all_metadata = []
        for jsonlfile in jsonlfiles:
            metadata = read_jsonl(jsonlfile)
            all_metadata.extend(metadata)

        return all_metadata

    def collate_fn(self, batch: List[dict]) -> Dict[str, List]:
        """
        Collate function to pad batch data for training.

        Args:
            batch (List[Dict]): List of data dictionaries to collate.

        Returns:
            Dict[str, List]: Collated batch data.
        """
        assert isinstance(batch, list)
        collate_batch = {}

        for k in batch[0].keys():
            collate_batch[k] = [b[k] for b in batch]

            return collate_batch


# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    config = {
        "jsonlfiles": {
            "train": "/home/xinshengwang/data/sparkvox/tmp/vctk/test.jsonl",
            "val": "/home/xinshengwang/data/sparkvox/tmp/vctk/test.jsonl",
        }
    }

    dataset = BaseDataset(config)
    dataloader = DataLoader(
        dataset, batch_size=12, num_workers=4, collate_fn=dataset.collate_fn
    )

    i = 0
    for batch in dataloader:
        i += 1
        print(f"itr {i}, batch_size:", len(batch["index"]))
