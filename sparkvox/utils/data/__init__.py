from sparkvox.utils.file import read_jsonl


def get_attribute_percentile_value(
    file_path: str = "sparkvox/utils/data/statistic_percentile.jsonl",
) -> dict:
    metadata = read_jsonl(file_path)
    return {meta["name"]: meta for meta in metadata}


PERCENTILE_MAP = get_attribute_percentile_value()

MEL_BOUNDARY_MALE = {
    "very_low": PERCENTILE_MAP["male_mel"]["percentile_5"],
    "low": PERCENTILE_MAP["male_mel"]["percentile_20"],
    "high": PERCENTILE_MAP["male_mel"]["percentile_70"],
    "very_high": PERCENTILE_MAP["male_mel"]["percentile_90"],
}

MEL_BOUNDARY_FEMALE = {
    "very_low": PERCENTILE_MAP["female_mel"]["percentile_5"],
    "low": PERCENTILE_MAP["female_mel"]["percentile_20"],
    "high": PERCENTILE_MAP["female_mel"]["percentile_70"],
    "very_high": PERCENTILE_MAP["female_mel"]["percentile_90"],
}

SPEED_BOUNDARY_ZH = {
    "very_low": PERCENTILE_MAP["zh_speed"]["percentile_5"],
    "low": PERCENTILE_MAP["zh_speed"]["percentile_20"],
    "high": PERCENTILE_MAP["zh_speed"]["percentile_80"],
    "very_high": PERCENTILE_MAP["zh_speed"]["percentile_95"],
}

SPEED_BOUNDARY_EN = {
    "very_low": PERCENTILE_MAP["en_speed"]["percentile_5"],
    "low": PERCENTILE_MAP["en_speed"]["percentile_20"],
    "high": PERCENTILE_MAP["en_speed"]["percentile_80"],
    "very_high": PERCENTILE_MAP["en_speed"]["percentile_95"],
}

LOUDNESS_BOUNDARY = {
    "very_low": PERCENTILE_MAP["loudness"]["percentile_5"],
    "low": PERCENTILE_MAP["loudness"]["percentile_20"],
    "high": PERCENTILE_MAP["loudness"]["percentile_80"],
    "very_high": PERCENTILE_MAP["loudness"]["percentile_95"],
}

PITCHVAR_BOUNDARY_MALE = {
    "very_low": PERCENTILE_MAP["male_std"]["percentile_5"],
    "low": PERCENTILE_MAP["male_std"]["percentile_20"],
    "high": PERCENTILE_MAP["male_std"]["percentile_80"],
    "very_high": PERCENTILE_MAP["male_std"]["percentile_95"],
}

PITCHVAR_BOUNDARY_FEMALE = {
    "very_low": PERCENTILE_MAP["female_std"]["percentile_5"],
    "low": PERCENTILE_MAP["female_std"]["percentile_20"],
    "high": PERCENTILE_MAP["female_std"]["percentile_80"],
    "very_high": PERCENTILE_MAP["female_std"]["percentile_95"],
}
