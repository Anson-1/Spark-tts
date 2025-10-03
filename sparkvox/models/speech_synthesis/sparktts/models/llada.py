import torch
import torch.nn as nn

from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparkvox.utils.file import load_config


class LLaDA(nn.Module):
    """`"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer_path: Path = None,
        token_num: int = None,
        infer: bool = False,
        **kargs
    ):
        super(LLaDA, self).__init__()

        self.token_num = token_num
        if not infer:
            self.init_model(model_name, tokenizer_path)

    def init_model(self, model_name: str, tokenizer_path) -> None:
        if tokenizer_path is None:
            tokenizer_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        self.model.resize_token_embeddings(self.token_num)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        return {"loss": loss, "logits": logits}

    def inference(self, text: str, temperature=0.8, top_k=50, top_p=0.95, **kwargs) -> str:
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=3000, do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response

    def chat_template(self, messages: List[Dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text


# test
if __name__ == "__main__":
    from torchmetrics.classification import MulticlassAccuracy

    top_1_accuracy = MulticlassAccuracy(
        200000, top_k=1, average="macro", ignore_index=-100
    )
    top_10_accuracy = MulticlassAccuracy(200000, average="macro", ignore_index=-100)

    model = LLaDA(
        model_name="/aifs4su/xinshengwang/model/public/huggingface/Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer_path="/aifs4su/xinshengwang/code/spark-tts/tokenizer/spark-tts-bicodec-pitch-energy-speech-tokenizer",
        token_num=200000
    )

    # infer
    prompt = "Can you tell me a joke?"
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    text = model.chat_template(messages)
    print("input text: \n", text)
    print("response: \n", model.inference(text))

    # forward

    model_inputs = model.tokenizer([text], return_tensors="pt").to(model.model.device)

    batch = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["input_ids"],
    }

    ouputs = model.forward(batch)

    loss = ouputs["loss"]
    logits = ouputs["logits"]
    print("loss", loss)

    print(
        "top1 acc:", top_1_accuracy(logits.transpose(1, 2), model_inputs["input_ids"])
    )
    print(
        "top10 acc:", top_10_accuracy(logits.transpose(1, 2), model_inputs["input_ids"])
    )