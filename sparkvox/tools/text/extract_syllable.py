import os
from typing import Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from sparkvox.utils.file import read_jsonl, write_jsonl
from sparkvox.tools.text.text2syllable import text2syllables


def get_syllable(
    index: str, text:str
) -> Tuple[float, float]:
    try:
        results = text2syllables(text)

        return {
            "index": index,
            "normalized_text": results['normalized_text'],
            "syllable_num": results['syllable_num'],
            "syllables": results['syllables']
        }
    except:
        return None


def main(jsonlfile, save_dir):
    data_name = save_dir.split("/")[-1]
    save_path = f"{save_dir}/syllable.jsonl"
    metadata = read_jsonl(jsonlfile)
    n_workers = min(128, os.cpu_count() * 2)
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for meta in tqdm(metadata, desc=data_name):
            index = meta["index"]
            if "text" not in meta:
                print(f'text missing in {meta["index"]}')
                continue
            text = meta["text"]
            futures.append(pool.submit(get_syllable, index, text))

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
        default="/aifs4su/mmdata/processed_data/spark-tts/NCSSD_C_EN/base.speaker.text.reindex.duration_max25s.jsonl",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/NCSSD_C_EN",
    )

    args = parser.parse_args()
    main(args.jsonlfile, args.save_dir)
