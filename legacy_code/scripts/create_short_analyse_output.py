import sys
import os

def process_file(file_path):
    try:
        if not os.path.isfile(file_path):
            print(f"Error: The file '{file_path}' does not exist.")
            return
        with open(file_path, 'r') as file:
            lines = file.readlines()

        target_line = "│   'Total': {"
        start_index = -1
        
        for i in range(len(lines) - 1, -1, -1):
            if target_line in lines[i]:
                start_index = i - 1  # Start one line before the found line
                break

        if start_index == -1:
            print(f"Error: The target line '{target_line}' was not found in the file.")
            return

        end_index = -1
        for i in range(start_index + 1, len(lines)):
            if lines[i].strip() == "}":
                end_index = i
                break

        if end_index == -1:
            print("Error: A line containing only '}' not found after the target line.")
            return

        output_file_path = os.path.join(os.path.dirname(file_path), "analyse_short_out.txt")

        with open(output_file_path, 'w') as output_file:
            output_file.writelines(lines[start_index:end_index + 1])
            if len(lines) > 1:
                output_file.write(lines[-1])
        print(f"Successfully created '{output_file_path}' with the selected lines.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
    else:
        process_file(sys.argv[1])
