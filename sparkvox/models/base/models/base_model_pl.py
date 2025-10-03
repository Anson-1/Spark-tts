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

import time
import torch
import pytorch_lightning as pl

from pathlib import Path
from hydra.utils import instantiate
from typing import Dict, List, Any, List, Tuple, Union
from torch import optim

from sparkvox.utils import log


class BaseModel(pl.LightningModule):
    """Base model for all models."""

    def __init__(self, config, **kwargs) -> None:
        """Initialize the base model."""
        super().__init__()

        self.config = config
        self.init_model()
        self.init_loss_functions()
        # save the config to the checkpoint
        self.save_hyperparameters()
        self.configure_optimizers()
        self.validation_step_outputs = []
        # self.automatic_optimization = False
        # self.init_non_trainable_model()

    def init_model(self) -> None:
        """Initialize the model."""
        # init your models with self.config
        pass

    def init_from_ckpt(self, path: Path, ignore_keys: List[str] = list()) -> None:
        """Initialize the model from a checkpoint."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))

        if ignore_keys:
            keys = list(checkpoint.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del checkpoint[k]
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=True)
        print(f"Restored from {path}")

    def configure_optimizers(self) -> Union[List, Tuple[List, List[dict]]]:
        """Configure optimizers with config."""
        self.optimizer = instantiate(
            self.config["optimizer"], params=self.model.parameters()
        )
        if self.config["lr_scheduler"] is not None:
            self.scheduler = instantiate(
                self.config["lr_scheduler"], optimizer=self.optimizer
            )
            lr_scheduler = {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return [self.optimizer], [lr_scheduler]
        else:
            return self.optimizer

    def log_stats(self, loss_dict: Dict[str, Any], prefix: str) -> None:
        """Log the statistics."""
        for k, v in loss_dict.items():
            if v is not None:
                self.log(f"{prefix}/{k}", v, on_epoch=True)

    def init_loss_functions(self) -> None:
        """Initialize the loss function."""
        pass

    def num_params(self) -> int:
        """Calculate the number of parameters in the model."""
        total_params = 0
        for p in self.model.parameters():
            if p.requires_grad:
                total_params += p.numel()
        return total_params

    def add_prefix_to_keys(self, d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add 'train_' or 'val_' to the keys in the loss dictionary."""
        return {f"{prefix}_{key}": value for key, value in d.items()}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Define the training step in the subclass."""
        raise NotImplementedError("training_step is not implemented")

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        """Define the validation step in the subclass."""
        pass

    def on_validation_start(self) -> None:
        """Ensure dropout is turned off and the model is in evaluation mode."""
        self.eval()

    @torch.inference_mode()
    def on_validation_end(self) -> None:
        """Ensure dropout is turned on and the model is in training mode."""
        self.train()

    def log_validation_outputs(self) -> None:
        """Aggregate losses from each batch in validation."""
        aggregated_loss = {}
        for loss_dict in self.validation_step_outputs:
            for key, value in loss_dict.items():
                if key not in aggregated_loss:
                    aggregated_loss[key] = []
                aggregated_loss[key].append(value)

        # Calculate the average for each loss metric
        for key, values in aggregated_loss.items():
            if isinstance(values[0], torch.Tensor):
                aggregated_loss[key] = torch.stack(values).mean()

        aggregated_loss = self.add_prefix_to_keys(aggregated_loss, 'agg')
        # Construct log message with losses and learning rates
        loss_msgs = ", ".join(
            f"{k}: {v:.6f}"
            for k, v in aggregated_loss.items()
            if isinstance(v, Union[float, torch.Tensor])
        )

        msg = f"Val | Step {self.global_step}, Epoch {self.current_epoch}, {loss_msgs}"

        if self.global_rank == 0:
            log.info(msg)

        # Log the aggregated average values at the end of the validation epoch
        self.log_dict(
            aggregated_loss,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # custom log
        self.validation_step_outputs = []

    def custom_log(self, loss_dict: Dict[str, Any], tag: str) -> None:
        """
        Logs the metrics if the current instance is on the main processing rank (rank 0).

        Args:
            loss_dict (Dict[str, float]): Dictionary containing loss metrics where keys are metric names
                                          and values are their corresponding loss values.
        """

        optimizers = self.optimizers()
        lr_dict = self.get_lr(optimizers)

        self.log_dict(
            loss_dict,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        loss_dict= {k.split('/')[-1]: v for k, v in loss_dict.items()}
        if self.global_rank == 0:
            # Construct log message with losses and learning rates
            loss_msgs = ", ".join(
                f"{k}: {v:.6f}"
                for k, v in loss_dict.items()
                if isinstance(v, Union[float, torch.Tensor])
            )
            lr_msgs = ", ".join(f"lr_{k}: {v:.2e}" for k, v in lr_dict.items())
            msg = f"{tag} | Step {self.global_step}, Epoch {self.current_epoch}, {loss_msgs}, {lr_msgs}"

            # Calculate elapsed and total estimated time if timestamps are available
            if hasattr(self, "start_time") and self.start_step is not None:
                elapsed_time = (time.time() - self.start_time) / 3600

                msg += f", Time: {elapsed_time:.1f} hour."

            # Start the timer at the first logging step
            else:
                self.start_step = self.global_step
                self.start_time = time.time()

            log.info(msg)

    def get_lr(
        self, optimizers: List[optim.Optimizer]
    ) -> Union[Dict[str, float], float]:
        """
        Get the learning rate from an optimizer or a dictionary of optimizers.

        Args:
            optimizers (List[optim.Optimizer]): A list of optimizers.
                of optimizer names to their corresponding optim.Optimizer objects.

        Returns:
            Union[Dict[str, float], float]: A dictionary mapping each optimizer to its current learning rate,
                or a single float representing the learning rate if a single optimizer is provided.
        """
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
            
        return {
            str(i): optimizer.param_groups[0]["lr"]
            for i, optimizer in enumerate(optimizers)
            if optimizer is not None
        }