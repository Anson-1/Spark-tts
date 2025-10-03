import re
import os

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from sparkvox.tools.text.text2syllable import TNEN
from sparkvox.tools.text.utils import get_word_list
from sparkvox.utils.file import read_jsonl, write_jsonl


def has_numbers(text):
    return bool(re.search(r"\d", text))


def count_words(words):
    clean_words = [t for t in words if re.search("\w", t)]
    return len(clean_words)


def replace_different_words(text_word_list, asr_word_list):
    # First record punctuation positions and marks in ASR text
    punct_positions = {}
    offset = 1
    for i, char in enumerate(asr_word_list):
        if not re.search("\w", char):
            if i == 0:
                continue
            punct_positions[i - offset] = char.replace(" ", "")
            offset += 1

    replacements = 0
    result_words = []

    # Process each word
    asr_clean = [t for t in asr_word_list if re.search("\w", t)]
    assert len(asr_clean) == len(text_word_list)

    for i in range(len(asr_clean)):
        text_word = text_word_list[i]
        asr_word = asr_clean[i]

        if asr_word.lower() != text_word.lower():
            # Keep the same case pattern as in ASR
            if asr_word.isupper():
                word = text_word.upper()
            elif asr_word.istitle():
                word = text_word.title()
            else:
                word = text_word.lower()
            replacements += 1
        else:
            word = asr_word
        if i in punct_positions:
            word = word + punct_positions[i]
        result_words.append(word)

    # Join words into text
    result = " ".join(result_words)

    return result, replacements


def process_jsonl(data):
    text = data["text"]
    asr = data["asr"]
    wer = data["wer"]
    meta = {"index": data["index"], "wer": wer, "org_text": text, "asr": asr}
    # Case 1: If WER is 0, keep ASR result
    if wer == 0:
        meta["text"] = asr

    else:
        if has_numbers(asr):
            asr = TNEN.normalize(asr)
        if has_numbers(text):
            text = TNEN.normalize(text)

        # Case 4: If word counts match
        text_words = get_word_list(text)
        asr_words = get_word_list(asr)

        text_words = [t for t in text_words if re.search("\w", t)]

        if count_words(text_words) == count_words(asr_words):
            new_asr, replacements = replace_different_words(text_words, asr_words)

            # If more than 3 replacements needed, skip this entry
            if replacements > 3:
                return

            meta["text"] = new_asr
        else:
            return

    return meta


def main(input_file, output_file):
    metadata = read_jsonl(input_file)
    new_metadata = []

    n_workers = min(128, os.cpu_count() * 2)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for meta in tqdm(metadata):
            futures.append(pool.submit(process_jsonl, meta))

        for f in tqdm(
            as_completed(futures), total=len(futures), desc=f"processing"
        ):
            meta = f.result()
            if meta is not None:
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
        "--save_path",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    main(args.jsonlfile, args.save_path)
