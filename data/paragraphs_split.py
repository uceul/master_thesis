import os
import json
import re
import shutil

def normalize_whitespace(text):
    """
    Normalize whitespace by:
    1. Replacing all Unicode whitespace characters with standard spaces.
    2. Removing all double spaces (collapsing them into single spaces).
    """
    # Replace all Unicode whitespace with standard spaces
    text = ''.join(' ' if c.isspace() else c for c in text)
    # Remove all double (or more) spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_directories(eval_path, train_path, input_dir, output_dir):
    # Step 1: Read JSON files into lists
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_json = json.load(f)
        eval_list = [normalize_whitespace(item['input']) for item in eval_json]
        print(len(eval_list))
        

    with open(train_path, 'r', encoding='utf-8') as f:
        train_json = json.load(f)
        train_list = [normalize_whitespace(item['input']) for item in train_json]
        print(len(train_list))

    in_eval_count = 0
    in_train_count = 0
    not_found_count = 0

    # Step 2: Iterate over subdirectories in the input directory
    for subdir in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir)

        if not os.path.isdir(subdir_path):
            continue  # Skip if not a directory

        content_file = os.path.join(subdir_path, 'content.txt')
        if not os.path.exists(content_file):
            print(f"Info: No content.txt found in {subdir}")
            continue

        # Step 3: Read the content of content.txt
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        content = normalize_whitespace(content)

        # Step 4: Check for substrings in eval and train lists
        in_eval = any(content in s for s in eval_list)
        in_train = any(content in s for s in train_list)

        if in_eval and in_train:
            print(f"Error: Subdirectory '{subdir}' is in both eval and train lists.")
            print(f"content: {content}")

        if in_eval:
            # Step 5: Copy subdirectory to output directory
            dest_path = os.path.join(output_dir, subdir)
            shutil.copytree(subdir_path, dest_path)
            print(f"Copied: {subdir} to {output_dir}")
            in_eval_count += 1
        elif in_train:
            in_train_count += 1
        else:
            not_found_count += 1
            if str(subdir) == "QAWPIO_clean":
                print(f"content: {content}")
            print(f"Info: Subdirectory '{subdir}' is not in eval or train lists.")

    print(f"Eval count: {in_eval_count}, train count: {in_train_count}, not found count: {not_found_count}")

# Example usage
# Replace these paths with actual ones before running
process_directories(
    eval_path="/home/iti/zn2950/home/ws/data/finetune/instruction/test.json",
    train_path="/home/iti/zn2950/home/ws/data/finetune/instruction/train.json",
    input_dir="/home/iti/zn2950/home/ws/data/synthesis_paragraphs",
    output_dir="/home/iti/zn2950/home/ws/data/eval_paragraphs"
)
