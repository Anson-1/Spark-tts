from datasets import load_from_disk
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

dataset_dir = "/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data_llada/m3ed"
split_name = "train_control_tts"
tokenizer_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts-llada/tokenizer"
max_print = 3
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

ds = load_from_disk(dataset_dir)
print(ds)
print("splits: ", list(ds.keys()))
train = ds[split_name]
print(train)


input_id = train[0]['labels']
decode = tokenizer.decode(input_id)
print(decode)
