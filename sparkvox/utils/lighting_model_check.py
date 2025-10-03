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

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


class CustomModelCheckpoint(ModelCheckpoint):
    """
    A custom ModelCheckpoint callback that saves a checkpoint after each training epoch without validation.
    
    """
    def __init__(self, **kargs):
        super().__init__(**kargs)
       
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        epoch = trainer.current_epoch + 1
        if self._should_save_on_train_epoch_end(trainer):
               monitor_candidates = self._monitor_candidates(trainer)
               if self.monitor not in monitor_candidates:
                   return
               filepath = self.format_checkpoint_name(monitor_candidates, 'epoch-' + str(epoch).zfill(2))
               self._save_checkpoint(trainer, filepath)

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int) -> None:
        if self.monitor is None or self.monitor not in trainer.callback_metrics:
            return
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)