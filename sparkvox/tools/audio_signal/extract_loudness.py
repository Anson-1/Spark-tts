import numpy as np

from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from sparkvox.utils.file import read_jsonl, write_jsonl
from sparkvox.utils.audio import load_audio
from sparkvox.tools.audio_signal.loudness import calculate_loudness


def get_loudness(
    index: str, audio_path: str, sample_rate: int
) -> Tuple[float, float]:
    try:
        audio = load_audio(audio_path, sampling_rate=sample_rate, volume_normalize=True)
        loudness = calculate_loudness(audio, sample_rate)

        return {
            "index": index,
            "loudness": round(float(loudness), 3)
        }
    except:
        return None


def main(jsonlfile, save_dir, sample_rate):
    data_name = save_dir.split("/")[-1]
    save_path = f"{save_dir}/loudness.jsonl"
    metadata = read_jsonl(jsonlfile)
    results = []
    with ProcessPoolExecutor(max_workers=256) as pool:
        futures = []
        for meta in tqdm(metadata, desc=data_name):
            index = meta["index"]
            wav_path = meta["wav_path"]
            futures.append(pool.submit(get_loudness, index, wav_path, sample_rate))

        for f in tqdm(
            as_completed(futures), total=len(futures), desc=f"{data_name} - processing"
        ):
            meta = f.result()
            if meta is not None:
                results.append(meta)

    write_jsonl(results, save_path)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonlfile",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/AISHELL-3/base.age.gender.accent.speaker.text.reindex.duration_max25s.jsonl",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/AISHELL-3",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)

    args = parser.parse_args()
    main(args.jsonlfile, args.save_dir, args.sample_rate)
