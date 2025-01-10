import re
import json
from matplotlib.ticker import PercentFormatter

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

import matplotlib.pyplot as plt
import numpy as np

def create_stacked_bars(model_errors, model_names):
    """
    Create stacked bar charts showing error distributions for multiple models.
    
    Parameters:
    - model_errors: List of dictionaries containing error data for each model
    - model_names: List of model names corresponding to the error data
    """
    parameters = list(model_errors[0].keys())  # Get parameters from first model
    n_models = len(model_errors)
    n_params = len(parameters)
    
    def get_combined_value(d, keys, default=0.0):
        """Sum up values for given keys if they exist in the dictionary"""
        return sum(d.get(k, default) for k in keys)
    
    # Create figure with sufficient width for multiple models
    # Increase figure height to accommodate rotated model labels
    fig, ax = plt.subplots(figsize=(max(12, n_models * 3), 9))
    
    # Add extra bottom margin for model labels
    plt.subplots_adjust(bottom=0.2)
    
    # Calculate positions for grouped bars
    x = np.arange(n_params)  # One position for each parameter
    width = 0.8 / n_models  # Width of bars, adjusted for number of models
    
    # Colors for error types
    error_colors = {
        'unit': '#3498db',
        'resolve': '#e74c3c',
        'found_in_text': '#f39c12',
        'wrong_no_additive': '#2ecc71',
        'other': '#95a5a6'
    }
    
    # Process data for each model
    for model_idx, (model_data, model_name) in enumerate(zip(model_errors, model_names)):
        # Calculate position offset for this model's group of bars
        offset = (model_idx - (n_models - 1) / 2) * width
        
        # Initialize arrays for each error category
        wrong_totals = []
        unit_vals = []
        resolve_vals = []
        found_in_text_vals = []
        wrong_no_additive_vals = []
        other_vals = []
        
        # Process each parameter
        for param in parameters:
            param_dict = model_data[param]
            
            # Get total wrong percentage for this parameter
            wrong_total = get_combined_value(param_dict, ['wrong'])
            wrong_totals.append(wrong_total)
            
            if wrong_total > 0:
                # Calculate each category's proportion of wrong answers
                unit = get_combined_value(param_dict, ['unit']) / wrong_total
                resolve = get_combined_value(param_dict, ['resolve_answer']) / wrong_total
                found_in_text = get_combined_value(param_dict, ['found_in_text']) / wrong_total
                wrong_no_additive = get_combined_value(param_dict, ['wrong_no_additive']) / wrong_total
                
                # Calculate 'other' as the remainder
                other = 1.0 - (unit + resolve + found_in_text + wrong_no_additive)
                
                # Append scaled values
                unit_vals.append(unit)
                resolve_vals.append(resolve)
                found_in_text_vals.append(found_in_text)
                wrong_no_additive_vals.append(wrong_no_additive)
                other_vals.append(max(0, other))
            else:
                # If no wrong answers, append zeros
                unit_vals.append(0)
                resolve_vals.append(0)
                found_in_text_vals.append(0)
                wrong_no_additive_vals.append(0)
                other_vals.append(0)
        
        # Create stacked bars for this model
        bar_x = x + offset
        bottom = np.zeros(n_params)
        
        # Only create legend for first model to avoid duplicates
        add_label = model_idx == 0
        
        # Create bars
        for error_type, values, color in [
            ('Unit Error', unit_vals, error_colors['unit']),
            ('Resolution Error', resolve_vals, error_colors['resolve']),
            ('Found in Text', found_in_text_vals, error_colors['found_in_text']),
            ('False Negative for Additive', wrong_no_additive_vals, error_colors['wrong_no_additive']),
            ('Other', other_vals, error_colors['other'])
        ]:
            p = ax.bar(bar_x, values, width * 0.9, bottom=bottom,
                      label=error_type if add_label else "", color=color)
            
            # Add value labels if segment is large enough
            for i, rect in enumerate(p):
                height = rect.get_height()
                if height > 0.02:  # Only show label if segment is > 2%
                    ax.text(rect.get_x() + rect.get_width()/2.,
                           rect.get_y() + height/2.,
                           f'{height:.1%}',
                           ha='center', va='center', fontsize=8)
            
            bottom += np.array(values)
    
    # Customize the plot
    ax.set_ylabel('Proportion of Wrong Answers')
    ax.set_title('Error Type Distribution in Wrong Answers')
    
    # Set x-axis labels with parameter names
    ax.set_xticks(x)
    ax.set_xticklabels(parameters)
    
            # Add model labels below each group of bars, only if there are errors
    for model_idx, (model_name, model_data) in enumerate(zip(model_names, model_errors)):
        offset = (model_idx - (n_models - 1) / 2) * width
        for param_idx, param in enumerate(parameters):
            # Check if there are any errors for this parameter/model combination
            if model_data[param].get('wrong', 0) > 0:
                # Position the text slightly to the left (-0.2 * width shifts it about 1cm left)
                ax.text(param_idx + offset - 0.1 * width, -0.05, model_name,
                       ha='right', va='top', fontsize=8, rotation=45)
    
    # Add legend at the top
    ax.legend(
        title='Error Types',
        bbox_to_anchor=(0.5, 1.15),
        loc='upper center',
        ncol=5
    )
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))

    plt.tight_layout()
    plt.savefig('error_distribution_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    accuracies_0 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/finetune/llama_32_1b/synthetic/llama_32_1b_instruct_full_synthetic_run_1_epoch_5_20250108_122553/analyse_short_out.txt")
    accuracies_1 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/finetune/llama_31_8b/synthetic/llama_31_8b_instruct_lora_synthetic_run_1_epoch_5_20250108_184649/analyse_short_out.txt")
    accuracies_2 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/finetune/llama_33_70b/synthetic/llama_33_70b_synthetic_instruction_lora_run_1_epoch_0_20250108_182957/analyse_short_out.txt")
    accuracies_3 = extract_parameters("/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/finetune/phi_3/synthetic/full_ft_run_1/phi_3_instruct_full_ft_synthetic_run_1_epoch_6_20250109_010535/analyse_short_out.txt")
    accuracies = [accuracies_0, accuracies_1, accuracies_2, accuracies_3]
    model_names = ["Llama 3.2 1B", "Llama 3.1 8B", "Llama 3.3 70B", "Phi 3 Mini"]  # Names of the mode # Names of the mode
    create_stacked_bars(accuracies, model_names)
