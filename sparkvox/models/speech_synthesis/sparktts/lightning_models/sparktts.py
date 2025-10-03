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

from hydra.utils import instantiate
from typing import Dict, Any
from omegaconf import DictConfig
from torchmetrics.classification import MulticlassAccuracy

from sparkvox.models.base.models.base_model_pl import BaseModel


class SparkTTS(BaseModel):
    """spark-tts model."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        super().__init__(config)
        pass

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass."""
        return self.model(batch)

    def init_model(self) -> None:
        """Initialize the model."""
        self.model = instantiate(self.config.llm)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        if len(self.validation_step_outputs) > 0:
            self.log_validation_outputs()
            
        input_ids = batch["input_ids"]
        print("input_ids:", input_ids.shape)
        loss_dict = {}
        output = self(batch)
        # caculate loss
        loss = output["loss"]

        if torch.isnan(loss):
            print(f"[Warning]: NaN loss detected, Skip back propagation.")
            return

        if (batch_idx + 1) % self.config.log_interval == 0:
            loss_dict["loss"] = loss
            self.custom_log(loss_dict, 'train')
        
        return loss

    @torch.inference_mode()
    def validation_step(self, batch: Dict[str, Any], batch_idx: int): 
        loss_dict = {}
        output = self(batch)

        loss_dict["val_loss"] = output["loss"]

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

    text = "<|im_start|>system You are spark-tts, created by WaveVortex. You are a helpful assistant.<|im_end|><|im_start|>user Can you tell me a joke?<|im_end|> <|im_start|>assistant"
    config = load_config("egs/speech_synthesis/spark-tts/config/spark-tts_qwen0.5b.yaml")
    model_config = config["model"]
    model = SparkTTS(model_config)

    model_inputs = model.model.tokenizer([text], return_tensors="pt").to(
        model.model.model.device
    )

    batch = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["input_ids"],
    }

    output = model(batch)
    print("loss", output["loss"])