import os
import torch
import soundfile as sf

from sparkvox.utils.file import load_config
from sparkvox.tools.tokenizer.audio_tokenizer.bicodec.audio_tokenizer import BiCodecTokenizer

device = torch.device("cuda:0")
config = load_config("egs/codec/bicodec/results/bicodec.24k/20250420_014312/config.yaml")

global_token_num = config['model']['generator']['speaker_encoder']['token_num']

tokenizer = BiCodecTokenizer(
        config_path="egs/codec/bicodec/results/bicodec.24k/20250420_014312/config.yaml",
        ckpt_path="egs/codec/bicodec/results/bicodec.24k/20250420_014312/ckpt/epoch=0010_step=110000.ckpt",
        wav2vec_model='pretrained_models/wav2vec2-large-xlsr-53',
        device=device,
    )

tokens = torch.load(
    "local/bicodec/m3ed/m3ed_Angry_0000000767.pt"
)

global_tokens = tokens[:global_token_num].unsqueeze(0).to(device)
semantic_tokens = tokens[global_token_num:].unsqueeze(0).to(device)

rec_wav = tokenizer.detokenize(global_tokens.unsqueeze(0), semantic_tokens)

sf.write(f"local/bicodec/check_code_rec.wav", rec_wav, config['datasets']['sample_rate'])