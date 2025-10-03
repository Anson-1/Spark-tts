from tqdm import tqdm
from sparkvox.utils.file import read_jsonl, write_jsonl


def get_text_dict(metadata):
    return {meta['index']: meta['text'] for meta in metadata}

def main(input_file, asr_file):
    metadata = read_jsonl(input_file)
    asr_results = read_jsonl(asr_file)
    text_dict = get_text_dict(asr_results)
    output_file = input_file.replace('.jsonl', '.upate_text.jsonl')
    new_metadata = []

    for meta in tqdm(metadata):
        index = meta['index']
        if index in text_dict:
            meta['text'] = text_dict[index]
            new_metadata.append(meta)

    write_jsonl(new_metadata, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonlfile",
        type=str,
        default="/aifs4su/mmdata/processed_data/spark-tts/NCSSD_C_EN/base.speaker.text.reindex.duration_max25s.jsonl",
    )

    parser.add_argument(
        "--asrfile",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args.jsonlfile, args.asrfile)
