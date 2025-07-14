import os
import re

def clean_files(base_dir):
    # Define replacements
    replacements = {
        "\n": " ",
        "\xa0": " ",
        "\u2005": " ",
        "\u2009": " "
    }

    for root, dirs, files in os.walk(base_dir):
        if "content.txt" in files:
            file_path = os.path.join(root, "content.txt")

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Perform replacements
                for old, new in replacements.items():
                    content = content.replace(old, new)

                # Write the cleaned content back to the file
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(content)

                print(f"Cleaned file: {file_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

def clean_single_file(file_path):
    # Define replacements
    replacements = {
        "\xa0": " ",
        "\u2005": " ",
        "\u2009": " "
    }
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Perform replacements
        for old, new in replacements.items():
            content = content.replace(old, new)

        # Write the cleaned content back to the file
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)

        print(f"Cleaned file: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


base_directory = "/home/iti/zn2950/home/ws/data/synthesis_paragraphs"  # Replace with your base directory.
#clean_files(base_directory)
single_file_path = "/home/iti/zn2950/home/ws/data/finetune/text_completion/dataset.txt"
clean_single_file(single_file_path)