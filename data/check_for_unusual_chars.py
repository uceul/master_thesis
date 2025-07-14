import os
import re

invisible_character_pattern = re.compile(r"[\x00-\x1F\x7F\xA0\u2000-\u200F\u2028-\u202F\u2060-\u206F]")


def find_unusual_characters(base_dir):
    # Define the characters we consider "normal".
    normal_characters = set(" \t\n\r")  # Spaces, tabs, newlines, and carriage returns.

    # Define a regex pattern for unusual invisible characters explicitly excluding normal ones.
    invisible_character_pattern = re.compile(r"[\x00-\x1F\x7F\xA0\u2000-\u200F\u2028-\u202F\u2060-\u206F]")

    files_with_unusual_chars = 0

    for root, dirs, files in os.walk(base_dir):
        if "content.txt" in files:
            file_path = os.path.join(root, "content.txt")

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    print(f"checking file: {file_path}")
                    content = file.read()

                # Search for unusual characters.
                matches = invisible_character_pattern.findall(content)

                if matches:
                    files_with_unusual_chars += 1
                    print(f"Warning: Found unusual invisible characters in {file_path}.")
                    for match in matches:
                        print(f"    Character(s): {repr(match)}")

            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Total files with unusual characters: {files_with_unusual_chars}")

def find_unusual_characters_single_file(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            print(f"checking file: {path}")
            content = file.read()

        # Search for unusual characters.
        matches = invisible_character_pattern.findall(content)

        if matches:
            print(f"Warning: Found unusual invisible characters in {path}.")
            for match in matches:
                if match != "\n":
                    print(f"    Character(s): {repr(match)}")

    except Exception as e:
        print(f"Error reading {path}: {e}")


# Example usage
base_directory = "/home/iti/zn2950/home/ws/data/eval_paragraphs"  # Replace with your base directory.
#find_unusual_characters(base_directory)
file = "/home/iti/zn2950/home/ws/data/synthetic_data/synthetic_instruct_dataset.json"
find_unusual_characters_single_file(file)