from torchtune.models.llama3 import llama3_tokenizer
from torchtune.datasets import instruct_dataset

tokenizer = llama3_tokenizer("/hkfs/work/workspace_haic/scratch/zn2950-llms/base_models/Llama-3.1-8B-Instruct/original/tokenizer.model")
ds = instruct_dataset(
    tokenizer=tokenizer,
    source="json",
    data_files="/home/iti/zn2950/home/ws/data/finetune/instruction/train.json",
    split="train"
)

for i in range(3):
    print(f"\nSample {i}:")
    # Extract input and output from original data
    original = ds.dataset[i]
    print("Original Input:", original["input"])
    print("Original Output:", original["output"])
    
    # Show tokenized version
    print("\nTokenized version:")
    print("Tokens:", ds[i]["tokens"][:50])
    print("Decoded:", tokenizer.decode(ds[i]["tokens"]))