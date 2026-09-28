"""
Extract the 1st and 3rd columns from kanct_drugs_first3cols.csv
and save the result as a new CSV file in the data/ directory.
"""

import csv
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "kanct_drugs_first3cols.csv"
OUTPUT_FILE = ROOT / "data" / "kanct_drugs_col1_col3.csv"


def extract_col1_and_col3(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with input_path.open(newline="", encoding="utf-8") as f_in, \
         output_path.open("w", newline="", encoding="utf-8") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        for row in reader:
            col1 = row[0] if len(row) > 0 else ""
            col3 = row[2] if len(row) > 2 else ""
            writer.writerow([col1, col3])
            row_count += 1

    print(f"Done. {row_count} rows processed.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    extract_col1_and_col3(INPUT_FILE, OUTPUT_FILE)
