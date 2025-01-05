import json
import matplotlib.pyplot as plt

def read_and_parse_data(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        content = (content
                  .replace('│', '')
                  .replace('\n', '')
                  .replace(' ', '')
                  .replace("'", '"'))
        data = json.loads(content)
        model_key = next(key for key in data.keys() if key != 'Total')
        return data[model_key]

def create_solvent_error_visualization(data, model_name):
    solvent_data = data['solvent']
    total = solvent_data['correct'] + solvent_data['wrong']
    
    # Calculate percentages
    correct_pct = (solvent_data['correct'] / total) * 100
    found_text_pct = (solvent_data['found_in_text'] / total) * 100
    found_text_unresolv_pct = (solvent_data['found_in_text_unresolvable'] / total) * 100
    remaining_pct = 100 - (correct_pct + found_text_pct + found_text_unresolv_pct)
    
    plt.figure(figsize=(12, 3))
    
    # Create stacked bars
    bars = plt.barh(y=['Solvent'], width=[correct_pct], color='royalblue', label='Correct')
    plt.barh(y=['Solvent'], width=[found_text_pct], left=[correct_pct], 
            color='gold', label='Found in Text')
    plt.barh(y=['Solvent'], width=[found_text_unresolv_pct], 
            left=[correct_pct + found_text_pct], 
            color='orange', label='Found in Text (Unresolvable)')
    plt.barh(y=['Solvent'], width=[remaining_pct], 
            left=[correct_pct + found_text_pct + found_text_unresolv_pct], 
            color='red', label='Wrong')
    
    # Add percentage labels on the bars
    percentages = [correct_pct, found_text_pct, found_text_unresolv_pct, remaining_pct]
    left_positions = [0, correct_pct, correct_pct + found_text_pct,
                     correct_pct + found_text_pct + found_text_unresolv_pct]
    
    for pct, left_pos in zip(percentages, left_positions):
        if pct >= 5:  # Only show label if segment is at least 5%
            plt.text(left_pos + pct/2, 0, f'{round(pct)}%',
                    ha='center', va='center', weight='bold')
    
    plt.xlim(0, 100)
    plt.title(f'Solvent Extraction Performance\nModel: {model_name}')
    plt.xlabel('Percentage')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return plt

def main(file_path, output_path=None):
    data = read_and_parse_data(file_path)
    
    with open(file_path, 'r') as file:
        content = file.read()
        content = content.replace('│', '').replace('\n', '').replace(' ', '').replace("'", '"')
        parsed = json.loads(content)
        model_name = next(key for key in parsed.keys() if key != 'Total')
    
    plt = create_solvent_error_visualization(data, model_name)
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file_path> [output_file_path]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    main(input_file, output_file)