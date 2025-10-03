# SparkVox

SparkVox is a training framework focused on speech generation, while also supporting a range of related speech tasks, including speaker attribute recognition, emotion recognition, audio codecs, and speech synthesis.

## Supported Tasks

- Speaker Attribute Recognition
    - Age prediction
    - Gender prediction
- Codec
    - BiCodec
    - BigCodec
- Speech Synthesis
    - SparkTTS

## Project Structure

- `bins`:
    - `train_pl`: The main training entry point for all tasks.
- `egs`:
    - `task` (e.g. codec, speech_synthesis): Example training scripts for each task.
- `sparkvox`
    - `models`: Model implementations for different tasks.
- `tools`: Utilities for data processing, model inference, and feature extraction.
- `utils`: Common utilities for tasks such as reading and processing audio files, as well as general training tools.


## Examples

- [BiCodec](./egs/codec/bicodec/README.md)
- [Age Predictor](./egs/speaker_attribute/age/readme.md)
- [Gender Predictor](./egs/speaker_attribute/gender/readme.md)
- [SparkTTS](./egs/speech_synthesis/spark-tts/README.md)


# Spark-tts
