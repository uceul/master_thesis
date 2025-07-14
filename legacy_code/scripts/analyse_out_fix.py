import os

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("_analyse_out.txt"):
                file_path = os.path.join(root, file)
                new_file_path = os.path.join(root, "analyse_short_out_complete.txt")
                process_file(file_path, new_file_path)

def process_file(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()

        # Find the first occurrence of the target line
        target_line = "│   'Total': {"
        for index, line in enumerate(lines):
            if target_line in line:
                start_index = max(0, index - 1)  # Include the line before the target line
                break
        else:
            print(f"Target line not found in {input_path}")
            return

        # Write the relevant lines to the new file
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines[start_index:])

        print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    # Specify the directory to process
    directory_to_process = "/home/iti/zn2950/home/ws/master_thesis/legacy_code/runs/finetune/phi_3/synthetic/lora_run_2"
    process_directory(directory_to_process)
