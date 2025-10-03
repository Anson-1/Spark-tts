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


from pathlib import Path

import os
import torch
import numpy as np

from typing import Any, Dict, Tuple
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from sparkvox.utils.file import load_config
from sparkvox.utils.audio import load_audio
from sparkvox.tools.tokenizer.base.abs_tokenizer import ABSTokenizer
from sparkvox.models.codec.BiCodec.modules.generator import Generator
from safetensors.torch import load_file
import hydra



class BiCodecTokenizer(ABSTokenizer):
    """BiCodec tokenizer

    Args:
        ckpt_path: path to the checkpoint
        config_path: path to the config
    """

    def __init__(
        self,
        config_path: Path = None,
        ckpt_path: Path = None,
        wav2vec_model: Path = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    ):
        super().__init__()
        """initialize the tokenizer"""
        config = load_config(config_path)
        self.config = config
        self.ssl_path = wav2vec_model
        self.device = device
        self.init_model(ckpt_path, config_path)

    # def init_model(self, ckpt_path: Path, config_path: Path):
    #     """initialize the model"""
    #     self.model = Generator.load_from_checkpoint(config_path, ckpt_path).to(
    #         self.device
    #     )
    #     self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
    #         self.ssl_path
    #     )
    #     self.feature_extractor = Wav2Vec2Model.from_pretrained(
    #         self.ssl_path
    #     ).to(self.device)
    #     self.feature_extractor.config.output_hidden_states = True

    def init_model(self, ckpt_path: Path, config_path: Path):
        """Build Generator from full training YAML and safetensors weights."""
        cfg = load_config(config_path)

        # 1. Instantiate every sub-module via Hydra
        mel_params = cfg["model"]["generator"]["mel_params"]
        encoder = hydra.utils.instantiate(cfg["model"]["generator"]["encoder"])
        decoder = hydra.utils.instantiate(cfg["model"]["generator"]["decoder"])
        quantizer = hydra.utils.instantiate(cfg["model"]["generator"]["quantizer"])
        speaker_encoder = hydra.utils.instantiate(cfg["model"]["generator"]["speaker_encoder"])
        prenet = hydra.utils.instantiate(cfg["model"]["generator"]["prenet"])
        postnet = hydra.utils.instantiate(cfg["model"]["generator"]["postnet"])

        self.model = Generator(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
            d_vector_train_start=cfg["model"]["generator"]["d_vector_train_start"],
        ).to(self.device)

        # 2. Load safetensors
        state_dict = load_file(str(ckpt_path))
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval() 

        # 3. Wav2Vec2 pieces
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.ssl_path)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(self.ssl_path).to(self.device)
        self.feature_extractor.config.output_hidden_states = True

    def get_ref_clip(self, wav: np.ndarray) -> np.ndarray:
        """Get reference audio clip for speaker embedding."""
        cfg = self.config["datasets"]
        ref_segment_length = (
            int(cfg["sample_rate"] * cfg["ref_segment_duration"])
            // cfg["latent_hop_length"]
            * cfg["latent_hop_length"]
        )
        wav_length = len(wav)

        if ref_segment_length > wav_length:
            # Repeat and truncate to handle insufficient length
            wav = np.tile(wav, (1 + ref_segment_length) // wav_length)

        return wav[:ref_segment_length]

    def process_audio(self, wav_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """load auido and get reference audio from wav path"""
        cfg = self.config["datasets"]
        wav_ref = load_audio(
            wav_path,
            sampling_rate=cfg["sample_rate"],
            volume_normalize=cfg["volume_normalize"],
        )

        wav = load_audio(
            wav_path,
            sampling_rate=cfg["sample_rate_for_ssl"],
            volume_normalize=cfg["volume_normalize"],
        )

        wav_ref = self.get_ref_clip(wav_ref)

        wav_ref = torch.from_numpy(wav_ref).unsqueeze(0).float()
        return wav, wav_ref

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        inputs = self.processor(
            wavs,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            output_hidden_states=True,
        ).input_values
        feat = self.feature_extractor(inputs.to(self.feature_extractor.device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    def tokenize_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """tokenize the batch of audio

        Args:
            batch:
                wavs (List[np.ndarray]): batch of audio
                ref_wavs (torch.Tensor): reference audio. shape: (batch_size, seq_len)

        Returns:
            semantic_tokens: semantic tokens. shape: (batch_size, seq_len, latent_dim)
            global_tokens: global tokens. shape: (batch_size, seq_len, global_dim)
        """
        feats = self.extract_wav2vec2_features(batch["wav"])
        batch["feat"] = feats
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def tokenize(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """tokenize the audio"""
        wav, ref_wav = self.process_audio(audio_path)
        feat = self.extract_wav2vec2_features(wav)
        batch = {
            "wav": torch.from_numpy(wav).unsqueeze(0).float().to(self.device),
            "ref_wav": ref_wav.to(self.device),
            "feat": feat.to(self.device),
        }
        semantic_tokens, global_tokens = self.model.tokenize(batch)

        return global_tokens, semantic_tokens

    def detokenize(
        self, global_tokens: torch.Tensor, semantic_tokens: torch.Tensor
    ) -> np.array:
        """detokenize the tokens to waveform
        
        Args:
            global_tokens: global tokens. shape: (batch_size, global_dim)
            semantic_tokens: semantic tokens. shape: (batch_size, latent_dim)

        Returns:
            wav_rec: waveform. shape: (batch_size, seq_len) for batch or (seq_len,) for single
        """
        wav_rec = self.model.detokenize(semantic_tokens, global_tokens)
        return wav_rec.detach().squeeze().squeeze().cpu().numpy()


# test
if __name__ == "__main__":
    import soundfile as sf


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BiCodecTokenizer(
        # config_path="egs/codec/bicodec/results/bicodec.24k/20250420_014312/config.yaml",
        # ckpt_path="egs/codec/bicodec/results/bicodec.24k/20250420_014312/ckpt/epoch=0010_step=110000.ckpt",
        config_path="/mnt/lsk_nas/anson/Spark/SparkVox/egs/codec/bicodec/config/bicodec.24k.yaml",
        ckpt_path="/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/Spark-TTS-0.5B/BiCodec/model.safetensors",
        wav2vec_model='pretrained_models/wav2vec2-large-xlsr-53',
        device=device,
    )
    wav_path = "egs/data/audios/m3ed/m3ed_Angry_0000000764.wav"

    global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)
    # import pdb; pdb.set_trace()
    wav_rec = tokenizer.detokenize(global_tokens, semantic_tokens)
    sf.write("local/wav_rec.wav", wav_rec, 24000)
