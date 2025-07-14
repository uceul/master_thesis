from torchtune.models.phi3 import phi3_mini_tokenizer
from torchtune.datasets import instruct_dataset

tokenizer = phi3_mini_tokenizer("/home/iti/zn2950/home/haicore_ws/base_models/Phi-3-mini-4k-instruct/tokenizer.model")
ds = instruct_dataset(
    tokenizer=tokenizer,
    source="json",
    data_files="/home/iti/zn2950/home/ws/data/finetune/instruction/train.json",
    split="train"
)

for i in range(3):
    print(f"\nSample {i}:")
    tokens = ds[i]["tokens"]
    labels = ds[i]["labels"]
    
    # Get only tokens where labels != -100
    training_indices = [idx for idx, label in enumerate(labels) if label != -100]
    training_tokens = [tokens[idx] for idx in training_indices]
    
    # Decode the training tokens
    decoded_training = [tokenizer.decode([t]) for t in training_tokens]
    
    print("\nOriginal data:")
    print("Input:", ds._data[i]["input"])
    print("Output:", ds._data[i]["output"])
    
    print("\nTokens being trained on (where labels != -100):")
    print("Token IDs:", training_tokens)
    print("Decoded tokens:", ", ".join(decoded_training))

    # New section: Print token ID and decoded value pairs
    print("\nToken-by-token decoding:")
    for token_id, decoded_token in zip(training_tokens, decoded_training):
        print(f"Token ID: {token_id:5d} -> '{decoded_token}'")

print(tokenizer.encode("\n"))
print(tokenizer.encode("\""))
print(tokenizer.encode('\"'))
print(tokenizer.encode("""
"""))