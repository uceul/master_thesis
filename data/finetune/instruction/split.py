from datasets import load_dataset
import json
from collections import defaultdict

# Load the dataset
dataset = load_dataset('json', data_files='converted.json')

# Group data by input text
grouped_data = defaultdict(list)
for item in dataset['train']:
    # Extract the input text without the question
    # Assuming the question is always at the end after some delimiter
    # You might need to adjust this depending on your exact format
    input_base = item['input'].split('\nOutput')[0].strip()  # Adjust split pattern if needed
    grouped_data[input_base].append(item)

# Convert to list of groups
groups = list(grouped_data.values())

# Calculate split index (10% for test)
import random
random.seed(42)  # For reproducibility
test_size = int(len(groups) * 0.1)

# Randomly shuffle the groups
random.shuffle(groups)

# Split the groups
test_groups = groups[:test_size]
train_groups = groups[test_size:]

# Flatten the groups back to individual examples
train_data = [item for group in train_groups for item in group]
test_data = [item for group in test_groups for item in group]

# Save to files
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

# Print some stats
print(f"Total number of input texts: {len(groups)}")
print(f"Number of input texts in train: {len(train_groups)}")
print(f"Number of input texts in test: {len(test_groups)}")
print(f"Total number of examples in train: {len(train_data)}")
print(f"Total number of examples in test: {len(test_data)}")
