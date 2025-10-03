# Copyright (c) 2025 Xinsheng Wang (w.xinshawn@gmail.com)
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


import re
import os
import hydra
import argparse
import torch
import soundfile
from tqdm import tqdm

from sparkvox.utils.file import load_config, read_jsonl
from sparkvox.models.speech_synthesis.sparktts.lightning_models.sparktts import SparkTTS
from sparkvox.models.speech_synthesis.sparktts.utils.token_parser import TASK_TOKEN_MAP

def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run tts inference.")

    # Add arguments to the parser
    parser.add_argument(
        "--audio_tokenizer_config", type=str, default="egs/speech_synthesis/spark-tts/config/audio_tokenizer/bicodec.yaml", help="Path to the configuration file."
    )
    parser.add_argument("--ckpt", type=str, default='/aifs4su/xinshengwang/code/spark-tts/sparkvox/egs/speech_synthesis/spark-tts/results/20250114_qwen_tts_under/20250117_0438/ckpt/epoch=0000_step=052000_loss=3.75.ckpt', help="Path to the .pt file.")
    parser.add_argument(
        "--save_dir", type=str, default='local/tmp/tts_u/others', help="Path to save generated audios"
    )
    parser.add_argument("--device", type=int, default=4, help="CUDA device number")
    parser.add_argument(
        "--wav_path", type=str, required=True, help="Path to JSONL file"
    )
    parser.add_argument(
        "--text", type=str, default='我非常荣幸的向大家介绍，由我们团队提出的可控语音合成技术。通过全新的语音编码方式，我们成功借助大语言模型实现了语音的可控生成，可以说是遥遥领先。'
    )
    parser.add_argument("--prompt_text", type=str, default='')
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling.")
    # Parse the arguments
    args = parser.parse_args()
    # Convert Namespace to dictionary
    args_dict = vars(args)

    print('gpu id', args.device)
    args_dict["device"] = torch.device(f"cuda:{args.device}")

    return args_dict

def inference_factory(cfg, args_dict):
    audio_tokenizer = hydra.utils.instantiate(cfg["audio_tokenizer"], device=args_dict['device'])
    
    # Check if the checkpoint is a safetensors file or a .ckpt file
    ckpt_path = args_dict['ckpt']
    if ckpt_path.endswith('.safetensors'):
        # Handle safetensors loading
        from sparkvox.models.speech_synthesis.sparktts.models.qwen import Qwen
        
        # Get the directory containing the safetensors file
        model_dir = os.path.dirname(ckpt_path)
        
        # Create a minimal inference wrapper that mimics SparkTTS but without the training components
        class InferenceModel:
            def __init__(self):
                # Load the Qwen model directly from the safetensors
                self.model = Qwen(model_name=model_dir, token_num=166000, infer=False)
            
            def inference(self, *args, **kwargs):
                return self.model.inference(*args, **kwargs)
            
            def forward(self, *args, **kwargs):
                return self.model.forward(*args, **kwargs)
        
        model = InferenceModel()
        model.model.to(args_dict['device'])
    else:
        # Handle .ckpt file loading (original method)
        model = SparkTTS.load_from_checkpoint(args_dict['ckpt'], map_location='cpu')
        model.model.to(args_dict['device'])
    
    return audio_tokenizer, model


def main(args_dict):
    cfg = load_config(args_dict["audio_tokenizer_config"])

    audio_tokenizer, model = inference_factory(cfg, args_dict)
    base_name = os.path.basename(args_dict['wav_path'])
    if not os.path.exists(args_dict['save_dir']):
        os.makedirs(args_dict['save_dir'])
    
    with torch.no_grad():
        for i in range(10):
            save_path = f'{args_dict['save_dir']}/{i}_{base_name}'
            global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(args_dict['wav_path'])
            global_tokens = "".join([f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()])
            if len(args_dict['prompt_text']) > 0:
                semantic_tokens = "".join([f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()])
                inputs = [
                TASK_TOKEN_MAP['tts'],
                "<|start_content|>",
                args_dict['prompt_text'],
                args_dict['text'],
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>", 
                semantic_tokens
            ]

            else:
                inputs = [
                        TASK_TOKEN_MAP['tts'],
                        "<|start_content|>",
                        args_dict['text'],
                        "<|end_content|>",
                        "<|start_global_token|>",
                        global_tokens,
                        "<|end_global_token|>",
                    ]
            inputs = "".join(inputs)
            print("input: ", inputs)
            predicts = model.model.inference(inputs, temperature=args_dict['temp'], top_k=args_dict['top_k'], top_p=args_dict['top_p'])
            print("predicts: ", predicts)
            
            # Let's also check what the model actually generated at the token level
            # This will help us see if there were stop tokens that got filtered out
            model_inputs = model.model.tokenizer([inputs], return_tensors="pt").to(model.model.model.device)
            print("model inputs: ", model_inputs)
            generated_ids = model.model.model.generate(**model_inputs, max_new_tokens=3000, do_sample=True, 
                                                      top_k=args_dict['top_k'], top_p=args_dict['top_p'], 
                                                      temperature=args_dict['temp'])
            generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]  # Remove input tokens
            print("Generated token IDs:", generated_ids.tolist()[:20], "... (showing first 20)")
            print("Last 10 generated tokens:", generated_ids.tolist()[-10:])
            
            # Decode WITH special tokens to see if there are stop tokens
            raw_decode = model.model.tokenizer.decode(generated_ids, skip_special_tokens=False)
            print("Raw decode (with special tokens):", raw_decode[-200:])  # Last 200 characters
            pred_semantic_ids = torch.tensor([int(token) for token in re.findall(r"\d+", predicts)]).long().unsqueeze(0)
            print(f"Shape of the final semantic ID tensor: {pred_semantic_ids.shape}")
            wav = audio_tokenizer.detokenize(global_token_ids.to(args_dict['device']), pred_semantic_ids.to(args_dict['device']))
            soundfile.write(save_path, wav, samplerate=16000)

if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)