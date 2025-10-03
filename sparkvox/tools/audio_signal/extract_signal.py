import numpy as np

from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from sparkvox.utils.audio import load_audio
from sparkvox.utils.file import read_jsonl, write_jsonl
from sparkvox.utils.audio import remove_silence_on_both_ends
from sparkvox.tools.audio_signal.rspl import get_rspl_mean_and_std
from sparkvox.tools.audio_signal.pitch import get_pitch_mean_and_std

import time


def get_speech_signal(
    index: str, audio_path: str, sample_rate: int
) -> Tuple[float, float]:
    try:
        audio = load_audio(audio_path, sampling_rate=sample_rate, volume_normalize=True)
        audio = remove_silence_on_both_ends(audio, sample_rate)
        speech_duration = len(audio) / sample_rate
        pitch, pitch_std = get_pitch_mean_and_std(audio, sample_rate)
        rspl, rspl_std = get_rspl_mean_and_std(audio, sample_rate)

        return {
            "index": index,
            "speech_duration": round(float(speech_duration), 3),
            "pitch": round(float(pitch), 3),
            "pitch_std": round(float(pitch_std), 3),
            "rspl": round(float(rspl), 3),
            "rspl_std": round(float(rspl_std), 3),
        }
    except:
        return None


def main(jsonlfile, save_dir, sample_rate):
    data_name = save_dir.split("/")[-1]
    save_path = f"{save_dir}/speech_signal.jsonl"
    metadata = read_jsonl(jsonlfile)
    results = []
    with ProcessPoolExecutor(max_workers=256) as pool:
        futures = []
        for meta in tqdm(metadata, desc=data_name):
            index = meta["index"]
            wav_path = meta["wav_path"]
            futures.append(pool.submit(get_speech_signal, index, wav_path, sample_rate))

        for f in tqdm(
            as_completed(futures), total=len(futures), desc=f"{data_name} - processing"
        ):
            meta = f.result()
            if meta is not None:
                results.append(meta)

    write_jsonl(results, save_path)


# def main(jsonlfile, save_dir, sample_rate):
#     get_speech_signal('index', '/aifs4su/mmdata/rawdata/speech/aishell3/train/wav/SSB1956/SSB19560481.wav', 16000)


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
