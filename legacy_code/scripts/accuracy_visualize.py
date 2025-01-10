import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import re
import json
from matplotlib.ticker import PercentFormatter

use_multiple = True

def extract_parameter_dict(content, param_name):
    # Find the position of the parameter
    param_pos = content.find(f"'{param_name.lower()}'")
    if param_pos == -1:
        return None
        
    # Find the opening brace after the parameter
    dict_start = content.find('{', param_pos)
    if dict_start == -1:
        return None
        
    # Keep track of nested braces
    brace_count = 1
    dict_end = dict_start + 1
    
    # Clean the content by removing vertical bars and extra whitespace
    cleaned_content = ''.join(line.strip().replace('│', '') for line in content[dict_end:].splitlines())
    
    # Find the matching closing brace in the cleaned content
    i = 0
    while brace_count > 0 and i < len(cleaned_content):
        if cleaned_content[i] == '{':
            brace_count += 1
        elif cleaned_content[i] == '}':
            brace_count -= 1
        i += 1
        
    if brace_count != 0:
        return None
    
    # Extract the dictionary string and convert to proper JSON format
    dict_str = '{' + cleaned_content[:i]
    
    # Clean up the string for JSON parsing
    dict_str = re.sub(r'\s+', ' ', dict_str)  # Replace multiple whitespace with single space
    dict_str = dict_str.replace("'", '"')  # Replace single quotes with double quotes
    
    try:
        return json.loads(dict_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing {param_name}: {e}")
        return None

def extract_parameters(filepath):
    with open(filepath, 'r') as file:
        content = file.read()
   
    parameters = {}
    param_names = ['Temperature', 'Time', 'Additive', 'Solvent']
   
    for param in param_names:
        param_dict = extract_parameter_dict(content, param.lower())
        if param_dict:
            parameters[param] = param_dict
   
    return parameters

def plot_accuracies(model_accuracies, model_names):
    """
    Plot accuracies grouped by model, with different colors for parameters.
    Parameters:
    - model_accuracies: List of dictionaries of accuracies for each model
    - model_names: List containing names of the models
    """
    parameters = list(model_accuracies[0].keys())
    n_models = len(model_accuracies)
    n_params = len(parameters)

    # Create positions for the bars
    x = np.arange(n_models)  # One position for each model
    width = 0.15  # Width of bars

    plt.figure(figsize=(14, 8))

    # Strong pastel colors for parameters
    param_colors = {
        'Temperature': '#87CEEB',  # Blue
        'Time': '#77DD77',         # Green
        'Solvent': '#FFB347',      # Cyan
        'Additive': '#FFB6C1'     # Lilac
    }

    # Plot bars for each parameter
    bars = []
    for i, param in enumerate(parameters):
        if use_multiple:
            param_values = [
                model[param]["correct"] + model[param].get("correct_multiple", 0)
                for model in model_accuracies
            ]
        else:
            param_values = [model[param]["correct"] for model in model_accuracies]
        offset = (i - (n_params - 1) / 2) * width  # Center the parameter bars for each model
        bar = plt.bar(x + offset, param_values, width * 0.9, 
                     label=param, color=param_colors[param])
        
        # Add value labels on the bars
        for j, rect in enumerate(bar):
            height = rect.get_height()
            accuracy_value = int(height * 100)
            if height > 0.10:
                plt.text(rect.get_x() + rect.get_width() / 2., height - 0.05,
                        f'{accuracy_value}',
                        ha='center', va='top', fontsize=10, color='black')
            else:
                plt.text(rect.get_x() + rect.get_width() / 2., height,
                        f'{accuracy_value}',
                        ha='center', va='bottom', fontsize=10, color='black')

    # Customize the plot
    plt.ylabel('Accuracy (%)', fontsize=14)

    # Set x-axis labels (model names)
    plt.xticks(x, model_names, fontsize=12, rotation=45, ha='right')

    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Set y-axis limits
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # Add legend at the top
    plt.legend(
        title='Parameters',
        title_fontsize=14,
        fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(parameters)
    )

    plt.tight_layout()

    # Save the plot
    plt.savefig('accuracy_comparison_plot_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()

accuracies_0 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/Vanilla/llama_32_1b/llama_32_1b_vanilla_new_20250108_032949/analyse_short_out.txt")
accuracies_1 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/Vanilla/llama_31_8b/llama_31_8b_vanilla_new_20250108_015630/analyse_short_out.txt")
accuracies_2 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/Vanilla/llama_33_70b/llama_33_70B_vanilla_evaluationset_20250107_173107/analyse_short_out.txt")
accuracies_3 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/Vanilla/phi_3_mini/phi_3_mini_vanilla_new_20250108_012229/analyse_short_out.txt")
accuracies = [accuracies_0, accuracies_1, accuracies_2, accuracies_3]
model_names = ["Llama 3.2 1B", "Llama 3.1 8B", "Llama 3.3 70B", "Phi 3 Mini"]  # Names of the mode
plot_accuracies(accuracies, model_names)

