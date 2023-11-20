import pandas as pd
import sys
from typing import List


def merge_csv_files(file1_path: str, file2_path: str, output_path: str) -> None:
    """
    Merge two CSV files based on the match between the values in the first two columns
    and output a new CSV file.

    Args:
        file1_path (str): Path to the first CSV file.
        file2_path (str): Path to the second CSV file.
        output_path (str): Path for the merged CSV file.
    """
    # Read CSV files into DataFrames
    colnames: List[str] = [
        "Column0",
        "Column1",
        "Column2",
    ]
    df1 = pd.read_csv(file1_path, names=colnames, header=None)
    df2 = pd.read_csv(file2_path, names=colnames, header=None)

    # Merge DataFrames based on the first two columns
    merged_df = pd.merge(df1, df2, on=["Column1", "Column2"])

    # Select columns for the output
    col_1_name: str = file1_path.split("/")[-1]
    col_2_name: str = file2_path.split("/")[-1]
    output_columns: List[str] = ["premise", "hypothesis", col_1_name, col_2_name]

    # Rename columns for clarity
    merged_df.columns: List[str] = [col_1_name, "premise", "hypothesis", col_2_name]
    # ["Column1", "Column2", "Column3_file1", "Column3_file2"]

    # Save the merged DataFrame to a new CSV file
    merged_df[output_columns].to_csv(output_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <file1_path> <file2_path> <output_path>")
        sys.exit(1)

    file1_path: str = sys.argv[1]
    file2_path: str = sys.argv[2]
    output_path: str = sys.argv[3]

    # Perform the merge and save the result
    merge_csv_files(file1_path, file2_path, output_path)

    print(f"Merged file saved to {output_path}")
