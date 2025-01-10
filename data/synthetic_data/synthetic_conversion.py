import pandas as pd
import json
from typing import List, Dict, Any
from pathlib import Path

# Global parameters
PROMPT = "Read the following Metal-Organic Framework (MOF) synthesis description and extract this information: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), one main solvent (no mixtures or ratios), one chemical additive ('None' if no additive present). Important: Do not use JSON or curly braces in your output, they are already provided for you and you do not need to generate them. Only output the extracted information and terminate strings with \""

SCHEMA = {
    "type": "object",
    "properties": {
        "additive": {"type": "string"},
        "solvent": {"type": "string"},
        "temperature": {"type": "number"},
        "temperature_unit": {"type": "string"},
        "time": {"type": "number"},
        "time_unit": {"type": "string"},
    },
}

# Order in which to extract fields
FIELD_ORDER = [
    "additive",
    "solvent", 
    "temperature",
    "temperature_unit",
    "time",
    "time_unit"
]

def is_number_field(field_name: str, schema: Dict[str, Any]) -> bool:
    """Check if a field is a number type in the schema."""
    return schema["properties"][field_name]["type"] == "number"

def create_partial_result(row: pd.Series, fields: List[str], schema: Dict[str, Any]) -> str:
    """Create the partial result JSON string up to the current field."""
    result = {}
    for field in fields:
        if pd.isna(row[field]):
            if field == "additive":
                result[field] = "None"
                continue
            else:
                raise ValueError(f"Field {field} is None")
        result[field] = row[field]
    
    # Convert to JSON string without the closing brace
    return json.dumps(result, ensure_ascii=False)[:-1]

def create_dataset(csv_path: str, schema: Dict[str, Any], prompt: str) -> List[Dict[str, str]]:
    """Create the input/output pairs dataset from the CSV file."""
    # Read CSV file
    df = pd.read_csv(csv_path, sep=';')
    
    # Initialize dataset
    dataset = []
    
    # Iterate over each row
    for _, row in df.iterrows():
        # Skip row if context is missing
        if pd.isna(row['context']):
            print(f"Skipping row {row['counter']} as it has no text")
            continue
        if pd.isna(row['solvent']):
            print(f"Skipping row {row['counter']} as it has no splvent")
            continue
        # Process each field in order
        for i, field in enumerate(FIELD_ORDER):
            # Create the input string
            previous_fields = FIELD_ORDER[:i]
            partial_result = create_partial_result(row, previous_fields, schema)
            
            # Add the current field key and appropriate punctuation
            if partial_result[-1] != "{":
                partial_result += ", "
            partial_result += f'"{field}": '
            
            # For string fields, add the opening quote
            if not is_number_field(field, schema):
                partial_result += '"'
            
            # Construct the full input
            input_text = (
                f"{prompt}\n{row['context']}\n"
                f"Output result in the following JSON schema format:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n"
                f"Result: {partial_result}"
            )
            
            # Get the output value and format it appropriately
            output = str(row[field])
            if not is_number_field(field, schema):
                if pd.isna(row[field]) and field == "additive":
                    output = "None"
                output += '"'  # Add closing quote for string fields
            
            # Add to dataset
            example = {
                "input": input_text,
                "output": output
            }
            dataset.append(example)
    
    return dataset

def main():
    # CSV file path - update this to your file path
    csv_path = "/home/iti/zn2950/home/ws/data/synthetic_data/synthetic_paragraphs.csv"
    
    # Create the dataset
    dataset = create_dataset(csv_path, SCHEMA, PROMPT)
    
    # Save to JSON file
    output_path = Path(csv_path).parent / "synthetic_instruct_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Created dataset with {len(dataset)} input/output pairs")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()