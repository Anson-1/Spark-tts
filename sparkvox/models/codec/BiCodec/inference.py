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


import soxr
import os
import hydra
import argparse
import torch
import numpy as np
from tqdm import tqdm
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sparkvox.utils.file import load_config, read_jsonl
from sparkvox.utils.audio import load_audio


def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run wav codec inference.")

    # Add arguments to the parser
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .pt file.")
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Path to save generated audios"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--jsonfile", type=str, required=True, help="Path to JSONL file"
    )
    parser.add_argument(
        "--wav2vec_model",
        type=str,
        default="/aifs4su/xinshengwang/model/wav2vec2-large-xlsr-53",
    )

    # Parse the arguments
    args = parser.parse_args()
    # Convert Namespace to dictionary
    args_dict = vars(args)

    args_dict["device"] = torch.device(f"cuda:{args.device}")

    return args_dict


def inference_factory(cfg, args_dict):
    model = hydra.utils.instantiate(cfg["model"]["generator"])
    model = model.load_from_checkpoint(
        args_dict["config"], args_dict["ckpt"], **args_dict
    )
    model.remove_weight_norm()
    return model.to(args_dict['device'])


def get_ref_clip(cfg, wav):
    """Get reference audio clip for speaker embedding."""
    if "ref_segment_duration" in cfg:
        ref_segment_length = (
            int(cfg["sample_rate"] * cfg["ref_segment_duration"])
            // cfg["latent_hop_length"]
            * cfg["latent_hop_length"]
        )
    else:
        ref_segment_length = cfg["ref_segment_frame"] * cfg["latent_hop_length"]

    wav_length = len(wav)

    if ref_segment_length > wav_length:
        # Repeat and truncate to handle insufficient length
        wav = np.tile(wav, (1 + ref_segment_length) // wav_length)

    return wav[:ref_segment_length]


def process_audio(wav_path, config):
    cfg = config['datasets']
    wav_raw = load_audio(
        wav_path,
        sampling_rate=cfg["sample_rate"],
        volume_normalize=cfg["volume_normalize"],
    )

    wav_ref = get_ref_clip(cfg, wav_raw)

    if cfg["sample_rate"] != cfg["sample_rate_for_ssl"]:
        wav_in = soxr.resample(
            wav_raw, cfg["sample_rate"], cfg["sample_rate_for_ssl"], quality="VHQ"
        )
    else:
        wav_in = wav_raw

    wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float().to(args_dict["device"])
    return wav_raw, wav_in, wav_ref

def main(args_dict):
    cfg = load_config(args_dict["config"])
    if "config" in cfg.keys():
        cfg = cfg["config"]

    processor = Wav2Vec2FeatureExtractor.from_pretrained(args_dict["wav2vec_model"])
    feature_extractor = Wav2Vec2Model.from_pretrained(args_dict["wav2vec_model"]).to(
        args_dict["device"]
    )
    feature_extractor.config.output_hidden_states = True

    model = inference_factory(cfg, args_dict)
    jsonfile = args_dict["jsonfile"]

    basename = os.path.basename(jsonfile).split(".")[0]
    save_dir = os.path.join(args_dict["save_dir"], basename)
    os.makedirs(save_dir, exist_ok=True)

    metadata = read_jsonl(jsonfile)

    for meta in tqdm(metadata, desc=basename):
        index = meta["index"]
        save_path = f"{save_dir}/{index}_rec.wav"
        if 'wav_path_obs' in meta:
            wav_path = meta["wav_path_obs"]
        else:
            wav_path = meta["wav_path"]

        wav_raw, wav_in, ref_wav = process_audio(wav_path, cfg)

        batch = {
            "wav": torch.from_numpy(wav_raw),
            "ref_wav": ref_wav}

        with torch.no_grad():
            inputs = processor(
                wav_in,
                sampling_rate=cfg["datasets"]["sample_rate_for_ssl"],
                return_tensors="pt",
                padding=True,
                output_hidden_states=True,
            ).input_values.to(args_dict["device"])
            feat = feature_extractor(inputs)
            feats_16 = feat.hidden_states[16]
            feats_14 = feat.hidden_states[14]
            feats_11 = feat.hidden_states[11]
            feats_mix = (feats_11 + feats_14 + feats_16) / 3
            batch["feat"] = feats_mix #.transpose(1, 2)

            outputs = model(batch)
        wav = outputs["recons"].squeeze().detach().cpu().numpy()

        sf.write(save_path, wav, samplerate=cfg['datasets']["sample_rate"])


if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)
