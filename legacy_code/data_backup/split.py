from datasets import load_dataset

dataset = load_dataset('json', data_files='converted.son')

# Split the dataset (e.g., 90% train, 10% validation)
split_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)

split_dataset['train'].to_json('train.json')
split_dataset['test'].to_json('val.json')