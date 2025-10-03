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
Mel classifier with pytorch lightning
"""

import torch
import torch.nn as nn
import torchaudio.transforms as TT

from hydra.utils import instantiate
from typing import Dict, Any
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from sparkvox.models.base.models.base_model_pl import BaseModel


class MelClassifier(BaseModel):
    """Mel classifier."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        super().__init__(config)

        mel_cfg = config["mel"]
        self.mel_transformer = TT.MelSpectrogram(
            mel_cfg["sample_rate"],
            mel_cfg["n_fft"],
            mel_cfg["win_length"],
            mel_cfg["hop_length"],
            mel_cfg["mel_fmin"],
            mel_cfg["mel_fmax"],
            n_mels=mel_cfg["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def init_model(self) -> None:
        """Initialize the model."""
        predictor = instantiate(self.config["predictor"])

        model = nn.ModuleDict({"predictor": predictor})
        self.model = model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        wav = batch["wav"]
        labels = batch["labels"]
        mel = self.mel_transformer(wav)
        logits = self.model["predictor"](mel.transpose(1, 2))
        output = {"labels": labels, "logits": logits}

        return output

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = {}
        output = self(batch)
        # caculate loss
        loss_dict["ce_loss"] = self.ce_loss(output["logits"], batch["labels"])
        loss = sum(
            [
                v * loss_dict[k]
                for k, v in self.config["loss_lambdas"].items()
                if k in loss_dict
            ]
        )

        if torch.isnan(loss):
            print(f"[Warning]: NaN loss detected, Skip back propagation.")
            return

        # caculate acc
        acc = self.acc_metric(output["logits"].detach(), batch["labels"])
        loss_dict["acc"] = acc
        loss_dict = self.add_prefix_to_keys(loss_dict, "train")

        self.log_dict(
            loss_dict,
            on_step=True,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if len(self.validation_step_outputs) > 0:
            self.log_validation_outputs()

        return loss

    @torch.inference_mode()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss_dict = {}
        output = self(batch)
        logits = output["logits"]
        labels = batch["labels"]

        loss_dict["val_ce_loss"] = self.ce_loss(output["logits"], batch["labels"])
        loss_dict["val_acc"] = self.acc_metric(logits.detach(), labels)
        self.validation_step_outputs.append(loss_dict)

        return loss_dict

    @torch.inference_mode()
    def inference(self, wav: torch.Tensor):
        if len(wav.shape) == 2:
            wav = wav.unsqueeze(0)
        logits = self.model["predictor"](wav)

        return logits

    def init_loss_functions(self):
        """Initialize the loss function and accuracy metric."""
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.acc_metric = MulticlassAccuracy(
            num_classes=self.config["predictor"]["embed_dim"], average="macro", top_k=1
        )

    def compute_pred_accuracy(self, logits, labels):
        return self.acc_metric(logits.detach(), labels)


# test
if __name__ == "__main__":
    from sparkvox.utils.file import load_config

    config = load_config(
        "egs/speaker_attribute/gender/config/mel_gender_predictor.yaml"
    )
    model = MelClassifier(config["model"])
    wav = torch.zeros([8, 16000])
    label = torch.zeros(8)
    batch = {"wav": wav, "labels": label}
    output = model(batch)
    print(output["logits"])
