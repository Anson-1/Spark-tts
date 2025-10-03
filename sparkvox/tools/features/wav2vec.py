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
Wav2Vec2.0 feature extractor for BiCodec
"""

import os 
import torch
import torch.distributed as dist
from tqdm import tqdm
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader 
from torch.utils.data.distributed import DistributedSampler
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from utils.file import read_jsonl
from utils.audio import load_audio


class AudioFeatureDataset(Dataset):
    def __init__(self, jsonlfile):
        super().__init__() 
        self.metadata = read_jsonl( jsonlfile )
        # self.meta_filter()

    def meta_filter(self):
        metadata= []
        for meta in tqdm(self.metadata, desc='filtering data'):
            wav_path = meta['wav_path']
            if not os.path.isfile(wav_path):
                continue

            metadata.append(meta)
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata) 

    def __getitem__(self, idx):
        meta = self.metadata[idx] 
        index = meta["index"]
        wav_path = meta['wav_path']
        try:
            wav = load_audio(wav_path, sampling_rate=16000, volume_normalize=True)
            wav = torch.from_numpy(wav).float()
            wav_length = len(wav)
        except:
            print(f"error loading {wav_path}")
            wav = None
            wav_length = 0
        
        return index, wav, wav_length

    def collate_fn(self,batch):
        max_length = max([b[-1] for b in batch])
        wavs = []
        input_lengths = torch.LongTensor(len(batch)).zero_()
        output_lengths = torch.LongTensor(len(batch)).zero_()
        indexs = [b[0] for b in batch]
        for i in range(len(batch)):
            _, waveform, wav_length = batch[i]
            wavs.append(waveform.numpy())
            input_lengths[i] = wav_length
            output_lengths[i] = wav_length // 320
        return indexs, wavs, input_lengths, output_lengths


def setup_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def extract(rank, args):
    setup_process(rank, args.world_size)
    device = torch.device(f"cuda:{rank}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_dir)
    feature_extractor = Wav2Vec2Model.from_pretrained(args.model_dir)
    feature_extractor.to(device)
    feature_extractor.config.output_hidden_states = True
    feature_extractor = DDP(feature_extractor, device_ids=[rank])
    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)

    dataset = AudioFeatureDataset(args.jsonlfile)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=12, collate_fn=dataset.collate_fn, sampler=sampler)
    
    save_dir_mix = os.path.join(args.save_dir,'wav2vec_mix')

    if not os.path.exists(save_dir_mix):
        os.makedirs(save_dir_mix, exist_ok=True)

    for batch in tqdm(dataloader): 
        indexs, wavs, input_lengths, output_lengths = batch
        with torch.no_grad():
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True, output_hidden_states=True).input_values.to(device)
            feat = feature_extractor(inputs)
            feats_16 = feat.hidden_states[16]
            feats_14 = feat.hidden_states[14]
            feats_11 = feat.hidden_states[11]
            feats_mix = (feats_11 + feats_14 + feats_16) / 3
        for i in range(len(indexs)):
            indx = indexs[i]
            length = output_lengths[i]
            feat_mix = feats_mix[i][:length].half().detach().cpu()
            if not torch.isnan(feat_mix).any():
                torch.save(feat_mix, f"{save_dir_mix}/{indx}.pt")       
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default='pretrained_models/wav2vec2-large-xlsr-53'
    )
    parser.add_argument(
        "--jsonlfile", type=str, default="/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test.jsonl"
    )
    parser.add_argument(
        "--save_dir", type=str, default="/aifs4su/xinshengwang/data/speech/vocoder/wav2vec"
    )
    parser.add_argument(
        '--data_dir', type=str, default='/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech'
    )
    parser.add_argument(
        '--wav_folder', type=str, default='wavs'
    )
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())

    args = parser.parse_args() m
    
    mp.spawn(extract, args=(args,), nprocs=args.world_size)

if __name__ == "__main__":
    main()