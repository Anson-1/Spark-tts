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
Wav codec with pytorch lightning
"""
import os
import torch
import soundfile
import numpy as np

from typing import List, Tuple
from hydra.utils import instantiate
from typing import Dict, Any
from omegaconf import DictConfig

from sparkvox.utils import log
from sparkvox.models.base.models.base_model_pl import BaseModel


class WavCodec(BaseModel):
    """Base model for wav codec. All wav codec models should inherit from this class."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        # disable automatic optimization to manually control the optimizer and scheduler
        # the optimizer and scheduler are defined in the configure_optimizers method
        self.automatic_optimization = False
        self.cached_demos = {}

    def init_model(self) -> None:
        """Initialize the model."""
        raise NotImplementedError("Subclass should implement this method.")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclass should implement this method.")

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        batch = self.update_batch_step(batch)
        output = self(batch)
        # get the optimizer and scheduler
        gen_optimizer, disc_optimizer = self.optimizers()
        gen_scheduler, disc_scheduler = self.lr_schedulers()

        disc_loss = self.discriminator_backward(output, disc_optimizer, disc_scheduler, batch_idx)
        loss_dict = self.generator_backward(output, gen_optimizer, gen_scheduler, batch_idx)

        loss_dict["disc_loss"] = disc_loss
        self.loss_dict = loss_dict

        if len(self.validation_step_outputs) > 0:
            self.log_validation_outputs()

        if (batch_idx + 1) % self.config.log_interval == 0:
            self.custom_log(self.loss_dict, 'train')

        if self.global_step % self.config.syn_interval == 0 :
            self.log_syn_wav(self.cached_demos)
            self.cached_demos = {}
        
        self.loss_dict = {}
 

    def discriminator_backward(
        self,
        inputs: Dict[str, Any],
        disc_optimizer: torch.optim.Optimizer,
        disc_scheduler: torch.optim.lr_scheduler._LRScheduler,
        batch_idx: int,
    ) -> None:
        # discriminator update
        disc_loss = self.model["discriminator"].discriminative_loss(inputs)
       
        if torch.isnan(disc_loss):
            print(f"[Warning]: NaN loss detected, Skip back propagation.")
            return

        disc_optimizer.zero_grad()
        self.manual_backward(disc_loss)
        self.clip_gradients(
            disc_optimizer,
            gradient_clip_val=self.config.optimizer.disc_grad_clip,
            gradient_clip_algorithm="norm",
        )
        disc_optimizer.step()
        disc_scheduler.step()

        return disc_loss
        # if (batch_idx + 1) % self.config.log_interval == 0:
        #     self.log_dict(
        #         {"train/disc_loss": disc_loss},
        #         on_step=True,
        #         prog_bar=True,
        #         logger=True,
        #         sync_dist=True,
        # )

    def generator_backward(
        self,
        inputs: Dict[str, Any],
        gen_optimizer: torch.optim.Optimizer,
        gen_scheduler: torch.optim.lr_scheduler._LRScheduler,
        batch_idx: int,
    ) -> None:
        # generator update
        self.set_requires_grad("discriminator", False)
        loss_dict = self.compute_generator_loss(inputs)
        gen_loss = loss_dict["gen_loss"]

        if torch.isnan(gen_loss):
            print(f"[Warning]: NaN loss detected, Skip back propagation.")
            return
        
        gen_optimizer.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(
            gen_optimizer,
            gradient_clip_val=self.config.optimizer.gen_grad_clip,
            gradient_clip_algorithm="norm",
        )
        gen_optimizer.step()
        gen_scheduler.step()

        loss_dict = self.add_prefix_to_keys(loss_dict, "train")

        self.set_requires_grad("discriminator", True)
        return loss_dict 

    @torch.inference_mode()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        batch = self.update_batch_step(batch)   
        output = self(batch)
        disc_loss = self.model["discriminator"].discriminative_loss(output)
        loss_dict = self.compute_generator_loss(output)
        loss_dict["disc_loss"] = disc_loss
        loss_dict = self.add_prefix_to_keys(loss_dict, "val")
        self.validation_step_outputs.append(loss_dict)

        self.cache_val_demos(output, batch['index'])

        return loss_dict

    @torch.inference_mode()
    def inference(self, wav: torch.Tensor) -> torch.Tensor:
        """Inference the model."""
        raise NotImplementedError("Subclass should implement this method.")

    def init_loss_functions(self):
        """Initialize the loss function and accuracy metric."""
        raise NotImplementedError("Subclass should implement this method.")

    def compute_generator_loss(
        self, batch: Dict[str, Any], output: Dict[str, Any]
    ) -> Dict[str, Any]:
        raise NotImplementedError("Subclass should implement this method.")

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        """Configure optimizers with config."""
        disc_params = self.model["discriminator"].parameters()
        gen_params = self.model["generator"].parameters()

        gen_optimizer = instantiate(
            self.config["optimizer"]['gen_optimizer'], params=gen_params
        )
        disc_optimizer = instantiate(
            self.config["optimizer"]["disc_optimizer"], params=disc_params
        )

        gen_scheduler = instantiate(
            self.config["optimizer"]["gen_lr_scheduler"], optimizer=gen_optimizer
        )
        disc_scheduler = instantiate(
            self.config["optimizer"]["disc_lr_scheduler"], optimizer=disc_optimizer
        )

        lr_gen_scheduler = {
            "scheduler": gen_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        lr_disc_scheduler = {
            "scheduler": disc_scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [gen_optimizer, disc_optimizer], [lr_gen_scheduler, lr_disc_scheduler]

    def set_requires_grad(self, model_name: str, flag: bool = True) -> None:
        for p in self.model[model_name].parameters():
            p.requires_grad = flag

    def update_batch_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return batch


    def cache_val_demos(self, outputs: Dict[str, torch.Tensor], indexs: List[str]):
        """Cache validation demo audio."""

        raw_wavs = outputs["audios"].squeeze(1).cpu().float().numpy()
        rec_wavs = outputs["recons"].squeeze(1).detach().cpu().float().numpy()

        for raw_wav, rec_wav, index in zip(raw_wavs, rec_wavs, indexs):
            self.cached_demos[index] = (raw_wav, rec_wav)
        
    def log_syn_wav(self, cached_demos: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """Save synthetic audio to local and log to tensorboard."""
        step = self.global_step 

        if len(cached_demos) == 0: 
            return
        
        sample_rate = self.config["sample_rate"]
        save_dir = os.path.join(self.config['log_dir'], f"val_results/{step}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for index, (raw_wav, rec_wav) in cached_demos.items():
            soundfile.write(f"{save_dir}/{index}_rec.wav", rec_wav, sample_rate)
            soundfile.write(f"{save_dir}/{index}_raw.wav", raw_wav, sample_rate)