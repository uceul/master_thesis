import pandas as pd
import json
import re
from typing import List, Dict, Any
from pathlib import Path

# Global parameters
INSTRUCTION = "Extract these synthesis conditions from the following Metal-Organic Framework (MOF) synthesis description: temperature (highest reaction temp, use 25°C if not specified), time (longest duration at highest temp), solvent (choose one main solvent, no mixtures or ratios), and one chemical additive (write 'None' if no additive present)"

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

# Labels file which unfortunately includes some paragraphs with "None" labels
LABELS_PATH = "/home/iti/zn2950/home/ws/master_thesis/legacy_code/mof_dataset_labeled_M.csv"  # Update this
# Original SynMOF_m label file to check for which paragraphs we have useful labels (!= "None")
SYNMOF_M_CSV_PATH = "/gfse/data/LSDF/lsdf01/lsdf/kit/iti/zn2950/ws/data/tobias/SynMOF_M_out.csv"

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
                raise ValueError(f"Field {field} is None for paragraph {row['paragraph_id']}")
        if field in ["solvent", "additive"]:
            # Handle solvent list - take first one or earliest appearing one
            syn_list = eval(row[field])
            earliest_pos = len(row['context'])
            selected_syn = syn_list[0]
            
            for syn in syn_list:
                pos = row['context'].find(syn)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    selected_syn = syn
            
            result[field] = selected_syn
        else:
            result[field] = row[field]  # json.dumps will handle the proper quoting
    
    # Convert to JSON string without the closing brace
    return json.dumps(result, ensure_ascii=False)[:-1]

def create_instruction_dataset(labels_path: str, schema: Dict[str, Any], instruction: str) -> List[Dict[str, str]]:
    """Create the instruction dataset from the labels file."""
    # Read CSV file
    df = pd.read_csv(labels_path, sep=';')
    df_control = pd.read_csv(SYNMOF_M_CSV_PATH, sep=';')
    # Initialize dataset
    dataset = []
    
    # Iterate over each row
    for _, row in df.iterrows():
        # Skip row if context is missing
        if pd.isna(row['context']):
            print(f"skipping row {row['paragraph_id']} as it has no text")
            continue

        # Skip row if paragraph_id is not in the control file
        if row['paragraph_id'] not in df_control['filename'].values:
            print(f"skipping row {row['paragraph_id']} as it is not in the control file")
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
                f"{row['context']}\n\n"
                f"Output result in the following JSON schema format:\n"
                f"{json.dumps(schema, ensure_ascii=False)}\n"
                f"Result: {partial_result}"
            )
            
            # Create the output
            if field in ["solvent", "additive"]:
                if pd.isna(row[field]):
                    if field == "additive":
                        output = "None"
                    else:
                        raise ValueError(f"Field {field} is None for paragraph {row['paragraph_id']}")
                else:
                    synonym_list = eval(row[field])

                    # Find earliest appearing checmical in context
                    earliest_pos = len(row['context'])
                    selected = synonym_list[0]                    
                    for syn in synonym_list:
                        pos = row['context'].find(syn)
                        if syn == "H2O":
                            # Dont want to find H2O if it is 5H2O
                            pos = len(row['context'])
                            match = re.search(r'(?<!\d)H2O', row['context'])
                            if match:
                                pos = match.start()
                        if pos != -1 and pos < earliest_pos:
                            earliest_pos = pos
                            selected = syn
                    
                    output = selected
            else:
                output = str(row[field])
                
            # Add to dataset
            example = {
                "instruction": instruction,
                "input": input_text,
                "output": output
            }
            dataset.append(example)
    
    return dataset

def main():
    # Create the dataset
    dataset = create_instruction_dataset(LABELS_PATH, SCHEMA, INSTRUCTION)
    
    # Save to JSON file
    output_path = Path(LABELS_PATH).parent / "instruction_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"Created instruction dataset with {len(dataset)} examples")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()