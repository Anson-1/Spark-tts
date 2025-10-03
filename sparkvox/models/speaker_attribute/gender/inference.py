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


import argparse
import torch
from tqdm import tqdm

from sparkvox.utils.audio import load_audio
from sparkvox.utils.file import read_jsonl
from sparkvox.utils.attribute_parser import AttributeParser
from sparkvox.models.speaker_attribute.gender.lightning_models.wavlm_gender_predictor import GenderPredictor


def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run tts inference.")

    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .pt file.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--jsonfile", type=str, required=True, help="Path to JSONL file"
    )
    # Parse the arguments
    args = parser.parse_args()
    # Convert Namespace to dictionary
    args_dict = vars(args)

    args_dict["device"] = torch.device(f"cuda:{args.device}")

    return args_dict

def inference_factory(args_dict):
    model = GenderPredictor.load_from_checkpoint(args_dict['ckpt'])
    model.model.to(args_dict['device'])
    model.eval()
    return model

def post_process(logits):
    gender = 'female' if logits[0] > logits[1] else 'male'
    return AttributeParser.gender(gender)

def main(args_dict):
    model = inference_factory(args_dict)
    jsonfile = args_dict["jsonfile"]

    metadata = read_jsonl(jsonfile)

    true_num, total_num = 0, 0
    for meta in tqdm(metadata):
        index = meta["index"]
        wav_path = meta["wav_path"]
        wav = load_audio(wav_path, sampling_rate=16000)
        try:
            with torch.no_grad():
                wav = torch.from_numpy(wav).to(args_dict['device'])
                logits = model.inference(wav.float())
                pred = post_process(logits[0])
                if 'gender' in meta:
                    if meta['gender'] == pred:
                        true_num += 1
                    total_num += 1
        except:
            continue

    print('gender acc', true_num / total_num)
                    
if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)



