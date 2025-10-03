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
spark-tts model with pytorch lightning
"""

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from typing import Dict, Any
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from sparkvox.models.base.models.base_model_pl import BaseModel


class SparkTTS(BaseModel):
    """spark-tts model."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        super().__init__(config)
        self.mask_token_id = 126336

    def forward_process(self, input_ids, eps=1e-3):
        b, l = input_ids.shape
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)

        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)
        return noisy_batch, masked_indices, p_mask

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass."""
        return self.model(**batch)

    def init_model(self) -> None:
        """Initialize the model."""
        self.model = instantiate(self.config.llm)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):

        input_ids = batch["input_ids"]
        print("input_ids:", input_ids.shape)
        print("pad_id:", self.config.pad_id)
        print("contains padding:", torch.any(input_ids == self.config.pad_id))

        if torch.rand(1) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            input_ids = input_ids[:, :random_length]

        noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
        attention_mask = (noisy_batch != self.config.pad_id).long()
        logits = self.model({"input_ids": noisy_batch, "attention_mask": attention_mask})['logits']

        token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        if torch.isnan(loss):
            print(f"[Warning]: NaN loss detected, Skip back propagation.")
            return

        if (batch_idx + 1) % self.config.log_interval == 0:
            # print("input_ids:", input_ids.shape)
            # print("pad_id:", self.config.pad_id)
            # print("contains padding:", torch.any(input_ids == self.config.pad_id))
            loss_dict = {"loss": loss}
            self.custom_log(loss_dict, 'train')
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int): 
        input_ids = batch["input_ids"]

        noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
        attention_mask = (noisy_batch != self.config.pad_id).long()
        logits = self.model({"input_ids": noisy_batch, "attention_mask": attention_mask})['logits']

        token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        loss_dict = {"val_loss": loss}
        self.validation_step_outputs.append(loss_dict)

        return loss_dict

    def init_loss_functions(self):
        """Initialize the loss function and accuracy metric."""
        self.acc_metric_top1 = MulticlassAccuracy(
            num_classes=self.config["llm"]["token_num"], average="macro", top_k=1, ignore_index=self.config['pad_id']
        )

        self.acc_metric_top10 = MulticlassAccuracy(
            num_classes=self.config["llm"]["token_num"], average="macro", top_k=10, ignore_index=self.config['pad_id']
        )

    def compute_pred_accuracy(self, logits, labels):
        acc_top1 = self.acc_metric_top1(logits.detach(), labels)
        acc_top10 = self.acc_metric_top10(logits.detach(), labels)
        return acc_top1, acc_top10


# test
if __name__ == "__main__":
    from sparkvox.utils.file import load_config

    text = "### Human: Can you tell me a joke?\n### Assistant:"
    config = load_config("egs/speech_synthesis/spark-tts/config/spark-tts_llada8b.yaml")
    model_config = config["model"]
    model = SparkTTS(model_config)

    model_inputs = model.model.tokenizer([text], return_tensors="pt").to(
        model.model.model.device
    )
    print(model_inputs)

    batch = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["input_ids"],
    }

    input_ids = batch["input_ids"]
    noisy_batch, masked_indices, p_mask = model.forward_process(input_ids)
    attention_mask = (noisy_batch != model.config.pad_id).long()
    logits = model.model({"input_ids": noisy_batch, "attention_mask": attention_mask})['logits']

    token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    print("loss", loss)
