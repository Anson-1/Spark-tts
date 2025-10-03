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


from __future__ import print_function
import hydra
import random
import torch
import numpy as np

from datetime import datetime
from typing import Tuple
from argparse import Namespace
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def add_model_args(parser):
    parser.add_argument("-c", "--config", required=True, help="config file")
    parser.add_argument(
        "-r",
        "--resume",
        default="0",
        type=str,
        help="resume checkpoint, last for the last checkpoint",
    )
    parser.add_argument("--accumluate", type=int, help='accumulate grad batches')
    parser.add_argument("--log_dir", required=True, help="save results dir")
    parser.add_argument("--checkpoint", help="checkpoint of models")
    parser.add_argument("--seed", default=1234)
    return parser

def add_log_args(parser):
    default_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument("--date", default=default_date, type=str, help="log date")
    parser.add_argument("--version", default=default_date, type=str, help="log version")
    parser.add_argument('--project', type=str, default='sparkvox')
    parser.add_argument('--enable_wandb', action='store_true')
    return parser


def add_device_args(parser):
    parser.add_argument("--nnodes", required=True, type=int)
    parser.add_argument("--nproc_per_node", required=True, type=int)
    return parser


def add_dataset_args(parser):
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="num of subprocess workers for reading",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Use pinned memory buffers used for reading",
    )
    parser.add_argument("--prefetch", default=100, type=int, help="prefetch number")
    parser.add_argument("--persistent_workers", default=True, type=bool)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    return parser


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def init_dataset_and_dataloader(
    args: Namespace, config: DictConfig, seed=777
) -> Tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    # total_batches = estimate_total_batches(args.utts_num, config['static']['batch_size'])
    train_dataset_sampler = hydra.utils.instantiate(config["dataloader"], config)
    val_dataset_sampler = hydra.utils.instantiate(
        config["dataloader"], config, mode="val"
    )

    if hasattr(train_data_loader, "sample"):
        train_dataset = train_dataset_sampler.sample()
        val_dataset = val_dataset_sampler.sample()

        train_data_loader = DataLoader(
            train_dataset,
            batch_size=None,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            generator=generator,
            prefetch_factor=args.prefetch,
        )
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=None,
            pin_memory=args.pin_memory,
            num_workers=args.num_workers,
            persistent_workers=True,
            generator=generator,
            prefetch_factor=args.prefetch,
        )

    return train_data_loader, val_data_loader
