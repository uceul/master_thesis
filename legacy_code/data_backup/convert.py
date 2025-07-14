import json

# Read the original file
with open('original.json', 'r') as f:
    # Load as JSON array if saved with orient='records'
    try:
        data = json.load(f)
    # If saved without proper JSON array format, load line by line
    except json.JSONDecodeError:
        data = [json.loads(line) for line in f]

# Create new data structure with combined fields
new_data = []
for item in data:
    # Replace any double newlines with single newlines in the input
    cleaned_input = item["input"].strip().replace("\n\n", "\n")
    
    new_item = {
        "input": item["instruction"] + "\n" + cleaned_input,
        "output": item["output"]
    }
    new_data.append(new_item)

# Save as proper JSON array with Unicode characters preserved
with open('converted.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, indent=2, ensure_ascii=False)