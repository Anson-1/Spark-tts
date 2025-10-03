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
Main script for training with pytorch_lightning
"""


import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning import seed_everything

from sparkvox.utils import log
from sparkvox.utils.file import load_config
from sparkvox.utils.train_utils import (
    add_model_args,
    add_log_args,
    add_device_args,
    add_dataset_args,
)
from sparkvox.utils.lightning_utils import (
    init_model,
    init_callbacks,
    init_log,
    init_dataloader,
    update_config,
    save_config,
    build_strategy,
    get_resume_path,
)


torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_log_args(parser)
    parser = add_device_args(parser)
    parser = add_dataset_args(parser)
    args = parser.parse_args()

    if args.version == "null":
        args.version = args.date

    if args.resume not in ["0", 0]:
        assert (
            args.version != args.date
        ), "if you want to resume training, you should specify a history version"

    args.log_dir = args.log_dir + "/" + args.version

    return args


def main():
    args = get_args()

    # set random seed
    seed_everything(args.seed, workers=True)

    # load config
    config = load_config(args.config)

    # update config based on args
    config = update_config(args, config)

    # init custom logger
    log.init(
        args.log_dir,
        date=args.date,
        enable_wandb=args.enable_wandb,
        project=args.project,
    )

    # init callbacks
    callbacks = init_callbacks(args, config["train"])

    # init logger
    logger = init_log(config["train"])

    # init model
    model = init_model(config["model"])

    # init dataloader
    train_dataloader, val_dataloader = init_dataloader(config["datasets"])

    # get resume ckpt path
    ckpt_path = get_resume_path(args)

    # init trainer
    trainer_config = config["train"]["trainer"]

    # build strategysss
    strategy, trainer_config = build_strategy(trainer_config)

    trainer = pl.Trainer(
        logger=logger, strategy=strategy, callbacks=callbacks, **trainer_config
    )

    # save config
    save_config(args, config, trainer)

    print(f"start training")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
