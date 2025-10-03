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

import json
import argparse
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sparkvox.utils.file import load_config
from sparkvox.utils.attribute_parser import AttributeParser
from sparkvox.tools.age_predictor.wav_dataset import WavDataset
from sparkvox.models.speaker_attribute.gender.lightning_models.wavlm_gender_predictor import GenderPredictor


def batch_to_cuda(batch: dict, rank):
    for k in batch:
        if type(batch[k]) is torch.Tensor:
            batch[k] = batch[k].cuda(rank)
    return batch


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def inference_factory(ckpt_path, device):
    model = GenderPredictor.load_from_checkpoint(ckpt_path)
    model.model.to(device)
    model.model.eval()
    return model


def post_process(logits):
    gender = 'female' if logits[0] > logits[1] else 'male'
    
    return AttributeParser.gender(gender)
    


def predict(rank, args):
    # setup_process(rank, args.world_size)
    device = torch.device(f"cuda:{rank}")
    config = load_config(args.config_path)
    print('init model')
    model = inference_factory(args.ckpt_path, device)
    print('model loaded')
    # model = DDP(model, device_ids=[rank]) 
    config["datasets"]["jsonlfiles_for_extract"] = args.jsonlfile
    dataset = WavDataset(config["datasets"], mode="val", extract_feat=True)
    # sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    print('init loader')
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=8, collate_fn=dataset.collate_fn, #sampler=sampler
    ) 

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # if rank == 0:
    #     iterator = tqdm(dataloader, 
    #                 total=len(dataloader),
    #                 desc=f'Processing',
    #                 disable=rank != 0)
    # else:
    #     iterator = dataloader
    print('start processing')
    f = open(f'{args.save_dir}/gender.jsonl', "w", encoding="utf-8")
    for batch in tqdm(dataloader):
        indexs = batch["index"]
        batch = batch_to_cuda(batch, rank)

        with torch.no_grad():
            logits = model.batch_inference(batch) #module
        
        for i in range(len(indexs)):
            index = indexs[i]
            pred = post_process(logits[i])
            meta = {
                'index': index,
                'pred': pred
            }
            json_str = json.dumps(meta, ensure_ascii=False) + "\n"
            f.write(json_str)
    f.close()



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonlfile",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/AGE_DATA/test.AISHELL-3.age.gender.accent.speaker.text.reindex.duration.jsonl",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/aifs4su/xinshengwang/code/spark-tts/sparkvox/egs/speaker_attribute/gender/config/wavlm_gender_ft.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/aifs4su/xinshengwang/code/spark-tts/sparkvox/egs/speaker_attribute/gender/results/20250105_wavlm_gender_ft/20250105_185201/ckpt/epoch=0001_step=018018_agg_val_acc=0.99.ckpt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/GENDER_DATA",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # mp.spawn(predict, args=(args,), nprocs=args.world_size)
    predict(args.device, args)


if __name__ == "__main__":
    main()
