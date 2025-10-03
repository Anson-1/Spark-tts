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
Train utils for pytorch_lightning
"""

import os
import hydra
import pytorch_lightning as pl

from argparse import Namespace
from typing import List, Tuple
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy, Strategy
from pytorch_lightning.callbacks import (
    OnExceptionCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)


def update_config(args: Namespace, config: DictConfig) -> DictConfig:
    config["train"]["trainer"]["num_nodes"] = args.nnodes
    config["train"]["trainer"]["devices"] = args.nproc_per_node
    if args.accumluate is not None:
        config["train"]["trainer"]["accumulate_grad_batches"] = args.accumluate
    config["train"]["logger"]["save_dir"] = f"{args.log_dir}/logs"
    config["datasets"]["dataloader"]["num_workers"] = args.num_workers
    config['model']['log_dir']=args.log_dir

    return config


def save_config(args: Namespace, config: DictConfig, trainer: pl.Trainer) -> None:
    # Save config to log_dir/config.yaml for inference and export

    if trainer.global_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        OmegaConf.save(config=config, f=f"{args.log_dir}/config.yaml")

        print(f"updated config is save to {args.log_dir}/config.yaml")


def init_model(model_config: DictConfig) -> pl.LightningModule:
    model = hydra.utils.instantiate(model_config, model_config)
    num_params = model.num_params()
    print(f"Model {model.__class__.__name__} has been built.")
    print(f"Model params: {num_params / 1000000:.02f}M")

    return model


def init_callbacks(args: Namespace, pl_config: DictConfig) -> List[pl.Callback]:
    callbacks = []
    if "callbacks" in pl_config:
        for name, config in pl_config.callbacks.items():
            print(f"init callbacks: {name}")
            if name == "model_checkpoint":
                config.dirpath = os.path.join(args.log_dir, "ckpt")
            callbacks.append(hydra.utils.instantiate(config))

    # progress bar
    callbacks.append(
        RichProgressBar()
        if pl_config.get("progress_bar", "tqdm") == "rich"
        else TQDMProgressBar()
    )

    # checkpoint on exception
    callbacks.append(
        OnExceptionCheckpoint(
            dirpath=os.path.join(args.log_dir, "ckpt"), filename="exception_stop"
        )
    )

    return callbacks


def init_log(pl_config: DictConfig) -> DictConfig:
    return hydra.utils.instantiate(pl_config.logger)


def build_strategy(config: DictConfig) -> Tuple[Strategy, DictConfig]:
    if config["strategy"] == "fsdp":
        strategy = FSDPStrategy(
            auto_wrap_policy={eval(config.strategy_params.wrap_block)},
            activation_checkpointing_policy={eval(config.strategy_params.wrap_block)},
            limit_all_gathers=config.strategy_params.limit_all_gathers,
            cpu_offload=config.strategy_params.cpu_offload,
        )
    elif config["strategy"] == "ddp":
        strategy = DDPStrategy(
            find_unused_parameters=config.strategy_params.find_unused_parameters
        )
    else:
        strategy = "auto"

    del config.strategy
    del config.strategy_params

    return strategy, config


def get_resume_path(args: Namespace) -> Path:
    ckpt_path = ""
    if args.resume not in [0, "0"]:
        ckpt_path = f"{args.log_dir}/ckpt/{args.resume}"

    elif args.checkpoint is not None:
        ckpt_path = args.checkpoint

    if os.path.isfile(ckpt_path):
        print(f"Resume from {ckpt_path}")

    else:
        ckpt_path = None

    return ckpt_path


def init_dataloader(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
    dataloader = hydra.utils.instantiate(config, config)
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    return train_dataloader, val_dataloader
