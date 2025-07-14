import os

# Define the base directory and output file
base_directory = "/gfse/data/LSDF/lsdf01/lsdf/kit/iti/zn2950/ws/data/synthesis_paragraphs/"
output_file = "dataset.txt"

file_count = 0
longest_line_length = 0

with open(output_file, "w", encoding="utf-8") as dataset:
    # Iterate through all subdirectories
    for subdir, _, files in os.walk(base_directory):
        # Check if 'content.txt' exists in the current subdirectory (should alawys exist!)
        if "content.txt" in files:
            content_path = os.path.join(subdir, "content.txt")
            try:
                with open(content_path, "r", encoding="utf-8") as content_file:
                    content = content_file.read()
                    # Replace newlines with spaces to ensure single-line output
                    single_line_content = content.replace("\n", " ").strip()
                    longest_line_length = max(longest_line_length, len(single_line_content))
                    dataset.write(single_line_content + "\n")
                    file_count += 1
            except Exception as e:
                print(f"Error reading {content_path}: {e}")
        else:
            print(f"Skipping {subdir} - content.txt not found.")

print(f"Dataset successfully created at {output_file}.")
print(f"Number of files processed: {file_count}")
print(f"Length of the longest line: {longest_line_length}")