import os
import pandas as pd
from pathlib import Path

def check_directory_csv_correspondence(base_dir, csv_path, filename_column):
    """
    Check correspondence between directory names and CSV filename entries.
    
    Parameters:
    base_dir (str): Path to the base directory containing subdirectories
    csv_path (str): Path to the CSV file
    filename_column (str): Name of the column containing filenames in the CSV
    
    Returns:
    tuple: (missing_from_csv, missing_directories)
        - missing_from_csv: directory names not found in CSV
        - missing_directories: CSV entries without corresponding directories
    """
    # Get all subdirectory names
    subdirs = [d.name for d in Path(base_dir).iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    csv_filenames = df[filename_column].tolist()
    print(f"Found {len(csv_filenames)} entries in CSV")
    
    # Convert to sets for efficient comparison
    subdir_set = set(subdirs)
    csv_set = set(csv_filenames)
    
    # Find directories that don't have corresponding CSV entries
    missing_from_csv = subdir_set - csv_set
    
    # Find CSV entries that don't have corresponding directories
    missing_directories = csv_set - subdir_set
    
    return missing_from_csv, missing_directories

def print_results(missing_from_csv, missing_directories):
    """Print the results in a formatted way"""
    print("\nResults:")
    print("-" * 50)
    
    print("\nDirectories missing from CSV:")
    if missing_from_csv:
        print(f"Found {len(missing_from_csv)} directories without CSV entries:")
        for item in sorted(missing_from_csv)[:10]:  # Show first 10 for brevity
            print(f"  - {item}")
        if len(missing_from_csv) > 10:
            print(f"  ... and {len(missing_from_csv) - 10} more")
    else:
        print("None - all directories have corresponding CSV entries")
    
    print("\nCSV entries missing directories:")
    if missing_directories:
        print(f"Found {len(missing_directories)} CSV entries without directories:")
        for item in sorted(missing_directories)[:10]:  # Show first 10 for brevity
            print(f"  - {item}")
        if len(missing_directories) > 10:
            print(f"  ... and {len(missing_directories) - 10} more")
    else:
        print("None - all CSV entries have corresponding directories")

# Example usage
if __name__ == "__main__":
    base_dir = "/pfs/work7/workspace/scratch/zn2950-llm_extraction/datasets/synthesis_paragraphs"
    csv_path = "/home/kit/iti/zn2950/ws/src/MOF_Literature_Extraction/Literature Extraction/Databases/SynMOF_A.csv"
    filename_column = "filename"  # Replace with your actual column name
    
    missing_from_csv, missing_directories = check_directory_csv_correspondence(
        base_dir, csv_path, filename_column
    )
    print_results(missing_from_csv, missing_directories)