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
import numpy as np
import torch.nn.functional as F

from sparkvox.utils.file import load_config, read_jsonl
from sparkvox.models.speech_synthesis.sparktts.lightning_models.sparktts_llada import SparkTTS
from sparkvox.models.speech_synthesis.sparktts.utils.token_parser import TASK_TOKEN_MAP

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, tokenizer, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        tokenizer: Tokenizer for decoding.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length


    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            print(f"Block {num_block + 1}/{num_blocks}, Step {i + 1}/{steps}:")
            print(tokenizer.batch_decode(x[:, prompt.shape[1]:], skip_special_tokens=False)[0])
            print("-" * 80)

    return x

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
    parser.add_argument("--steps", type=int, default=128, help="Sampling steps.")
    parser.add_argument("--gen_length", type=int, default=128, help="Generated answer length.")
    parser.add_argument("--block_length", type=int, default=128, help="Block length.")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="Unsupervised classifier-free guidance scale.")
    parser.add_argument("--remasking", type=str, default='low_confidence', help="Remasking strategy.")
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
        from sparkvox.models.speech_synthesis.sparktts.models.llada import LLaDA
        
        # Get the directory containing the safetensors file
        model_dir = os.path.dirname(ckpt_path)
        
        # Create a minimal inference wrapper that mimics SparkTTS but without the training components
        class InferenceModel:
            def __init__(self):
                # Load the LLaDA model directly from the safetensors
                self.model = LLaDA(model_name=model_dir, token_num=166000, infer=False)

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
            # print("input: ", inputs)
            input_ids = model.model.tokenizer([inputs], return_tensors="pt").to(model.model.model.device)['input_ids']
            predicts = generate(model.model.model, input_ids, model.model.tokenizer, steps=128, gen_length=128, block_length=32, temperature=args_dict['temp'], cfg_scale=0., remasking='low_confidence')
            predicts = model.model.tokenizer.batch_decode(predicts[:, input_ids.shape[1]:], skip_special_tokens=False)[0]
            print(f"Length of the decoded string (characters): {len(predicts)}")
            
            pred_semantic_ids = torch.tensor([int(token) for token in re.findall(r"\d+", predicts)]).long().unsqueeze(0)
            print(f"Shape of the final semantic ID tensor: {pred_semantic_ids.shape}")
            print(f"Predicted Semantic IDs: {pred_semantic_ids}")

            wav = audio_tokenizer.detokenize(global_token_ids.to(args_dict['device']), pred_semantic_ids.to(args_dict['device']))
            soundfile.write(save_path, wav, samplerate=16000)

if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)