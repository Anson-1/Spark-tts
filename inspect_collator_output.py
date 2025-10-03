
import sys
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add project root to sys.path to allow importing sparkvox
# This allows us to run this script from the project root directory
project_root = "/mnt/lsk_nas/anson/Spark/SparkVox"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import the custom collator
from sparkvox.models.speech_synthesis.base.dataloaders.data_collator_for_lm import DataCollatorForCausalLM

def inspect_data():
    """
    Loads the dataset, passes a batch through the collator, and prints the output.
    """
    # 1. Define paths
    dataset_path = "/mnt/lsk_nas/anson/Spark/SparkVox/local/sparktts/train_data_dllm/m3ed"
    tokenizer_path = "/mnt/lsk_nas/anson/Spark/SparkVox/pretrained_models/spark-tts/tokenizer"
    BATCH_SIZE = 2 # Use a small batch size for easy inspection

    # 2. Load dataset and tokenizer
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure you have run the prepare_train.py script correctly.")
        return

    # Let's inspect the 'train_tts' split
    split_to_inspect = "train_tts"
    if split_to_inspect not in dataset:
        print(f"ERROR: Split '{split_to_inspect}' not found in the dataset.")
        print(f"Available splits: {list(dataset.keys())}")
        return
    subset = dataset[split_to_inspect]

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 3. Instantiate the Data Collator
    print("Instantiating DataCollatorForCausalLM...")
    data_collator = DataCollatorForCausalLM(
        tokenizer_path=tokenizer_path,
        padding_side="left",
        mlm=False,
        return_tensors="pt",
        max_length=1500
    )

    # 4. Create a DataLoader
    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
        shuffle=False # Important for reproducibility
    )

    # 5. Get a single batch
    print("\nFetching one batch from the DataLoader...")
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("ERROR: The dataloader is empty. The dataset split might have no data.")
        return


    # 6. Print the results
    print("\n--- Inspecting the Collator's Output Batch ---")
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Shape of input_ids:      {batch['input_ids'].shape}")
    print(f"Shape of attention_mask: {batch['attention_mask'].shape}")
    print(f"Shape of labels:         {batch['labels'].shape}")

    print("\n--- Looking at the first example in the batch ---")

    # Get the first example from the batch
    input_ids_example = batch['input_ids'][0]
    labels_example = batch['labels'][0]

    print(f"\nFirst 70 token IDs from input_ids: \n{input_ids_example[:70]}...")
    print(f"\nFirst 70 token IDs from labels: \n{labels_example[:70]}...")

    # Find where the masking stops
    unmasked_labels_start = -1
    for i, label_id in enumerate(labels_example):
        if label_id != -100:
            unmasked_labels_start = i
            break

    print(f"\nLabel masking (-100) stops at index: {unmasked_labels_start}")
    if unmasked_labels_start != -1:
        print(f"The first 20 unmasked labels are: \n{labels_example[unmasked_labels_start:unmasked_labels_start+20]}...")

    # Decode for human-readable output
    print("\n--- Decoded Tokens for the first example ---")
    print(f"Padding token ID is: {tokenizer.pad_token_id}")
    
    # For decoding labels, we need to replace -100 so it doesn't error out
    labels_for_decoding = labels_example.clone()
    labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.decode(labels_for_decoding, skip_special_tokens=False)
    print(f"\nDecoded labels (with -100 replaced by PAD token):\n{decoded_labels}\n")
    
    decoded_inputs = tokenizer.decode(input_ids_example, skip_special_tokens=False)
    print(f"Decoded input_ids:\n{decoded_inputs}\n")


    # --- NEW SECTION TO MANUALLY FIND SEMANTIC TOKENS ---
    print("\n--- Manually Finding the Semantic Tokens ---")

    # The semantic tokens should start right after the <|start_semantic_token|>
    start_semantic_token = "<|start_semantic_token|>"
    try:
        start_semantic_token_id = tokenizer.convert_tokens_to_ids(start_semantic_token)
        print(f"The token ID for '{start_semantic_token}' is: {start_semantic_token_id}")

        # Search for this token in the first example's input_ids
        # The input_ids contains the full concatenated sequence
        input_ids_example = batch['input_ids'][0]
        indices = (input_ids_example == start_semantic_token_id).nonzero(as_tuple=True)[0]

        if len(indices) > 0:
            start_semantic_index = indices[0].item()
            print(f"Found '{start_semantic_token}' at index: {start_semantic_index} in the full input_ids.")

            # The actual semantic tokens start at the next index
            actual_semantic_tokens_start_index = start_semantic_index + 1
            # Let's grab a slice of 30 tokens
            semantic_token_ids = input_ids_example[actual_semantic_tokens_start_index : actual_semantic_tokens_start_index + 30]

            print(f"\nThe *actual* semantic tokens therefore start at index {actual_semantic_tokens_start_index}.")
            print(f"The first 30 semantic token IDs are:\n{semantic_token_ids}")

            # Decode these tokens
            decoded_semantic_tokens = tokenizer.decode(semantic_token_ids)
            print(f"\nDecoded semantic tokens:\n{decoded_semantic_tokens}")

        else:
            print(f"Could not find the '{start_semantic_token}' in the input_ids.")

    except Exception as e:
        print(f"An error occurred while trying to find semantic tokens: {e}")
        print("This might be because '<|start_semantic_token|>' is not in the tokenizer's vocabulary.")


    print("\n--- End of Inspection ---")

if __name__ == "__main__":
    inspect_data()
