import csv

def count_empty_additive_rows(file_path):
    empty_count = 0

    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            if not row['additive1'].strip():
                empty_count += 1

    return empty_count

# Replace 'your_file.csv' with the path to your actual file
file_path = '/home/iti/zn2950/home/ws/data/results/SynMOF_A_out.csv'
empty_additive_count = count_empty_additive_rows(file_path)
print(f"Number of rows with empty 'additive' column: {empty_additive_count}")
