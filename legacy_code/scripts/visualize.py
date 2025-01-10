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

def calculate_percentages(data):
    percentages = {}
    for param, values in data.items():
        total = values['correct'] + values['wrong']
        if total > 0:
            percentages[param] = (values['correct'] / total) * 100
    return percentages

def create_visualization(data, model_name):
    percentages = calculate_percentages(data)
    plt.figure(figsize=(10, 6))
    parameters = list(percentages.keys())
    values = list(percentages.values())
    
    bars = plt.bar(parameters, values)
    plt.title(f'Correct Answers Percentage by Parameter\nModel: {model_name}')
    plt.ylabel('Correct Answers (%)')
    plt.ylim(0, 100)
    
    for bar in bars:
        height = bar.get_height()
        percentage = f'{round(height)}%'
        if height > 50:
            plt.text(bar.get_x() + bar.get_width()/2., height - 5,  # Position near top inside bar
                    percentage,
                    ha='center', va='top',
                    weight='bold', color='black')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    percentage,
                    ha='center', va='bottom',
                    weight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def main(file_path, output_path=None):
    data = read_and_parse_data(file_path)
    
    with open(file_path, 'r') as file:
        content = file.read()
        content = content.replace('│', '').replace('\n', '').replace(' ', '').replace("'", '"')
        parsed = json.loads(content)
        model_name = next(key for key in parsed.keys() if key != 'Total')
    
    plt = create_visualization(data, model_name)
    
    if output_path:
        plt.savefig(output_path)
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