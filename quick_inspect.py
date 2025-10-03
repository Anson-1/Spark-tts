from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("/mnt/lsk_nas/anson/Spark/Open-dLLM/scripts/sparktts_tokenizer/tokenizer.json")
print("Vocab size:", tokenizer.get_vocab_size())