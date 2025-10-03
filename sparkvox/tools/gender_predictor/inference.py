import json
import argparse
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import BasePredictionWriter
from torch.utils.data import DataLoader
from sparkvox.utils.file import load_config
from sparkvox.utils.attribute_parser import AttributeParser
from sparkvox.models.speaker_attribute.age.utils.age_parser import age_parser_rev
from sparkvox.tools.age_predictor.wav_dataset import WavDataset
from sparkvox.models.speaker_attribute.gender.lightning_models.wavlm_gender_predictor import GenderPredictor


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_path = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        with open(self.output_path, 'a') as f:
            for prediction in predictions:
                for i in range(len(prediction[0])):
                    index = prediction[0][i]
                    pred = prediction[1][i]
                    pred = post_process(pred)
                    meta = {
                        'index': index,
                        'pred': pred
                    }

                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def post_process(logits):
    gender = 'female' if logits[0] > logits[1] else 'male'
    
    return AttributeParser.gender(gender)


# Main function refactored for PyTorch Lightning with multi-GPU support
def main():
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
        "--save_path",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/GENDER_DATA/gender.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())  # Multi-GPU option
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())  # World size for DDP

    args = parser.parse_args()

    print('processing data', args.jsonlfile)
    config = load_config(args.config_path)
    config["datasets"]["jsonlfiles_for_extract"] = args.jsonlfile

    # Create dataset and dataloader
    dataset = WavDataset(config["datasets"], mode="val", extract_feat=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset.collate_fn) 

    # Initialize the model
    model = GenderPredictor.load_from_checkpoint(args.ckpt_path)
    pred_writer = CustomWriter(output_dir=args.save_path, write_interval="epoch")
    trainer = pl.Trainer(accelerator="gpu", strategy="ddp", devices=-1, callbacks=[pred_writer])

    trainer.predict(model, dataloaders=dataloader)

if __name__ == "__main__":
    main()
