import pandas as pd
import sys


def filter_csv_rows(input_path: str, output_path: str) -> None:
    """
    Filter rows of a CSV file based on specified conditions and save the result to a new CSV file.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path for the output CSV file.
    """
    # Read CSV file into a DataFrame
    df = pd.read_csv(input_path)

    # Get the last two column names dynamically
    last_two_columns = df.columns[-2:]

    # Apply the conditions to filter rows
    filtered_df = df[
        ((df[last_two_columns[0]] == 1) & (df[last_two_columns[1]] == "ENTAILMENT"))
        | ((df[last_two_columns[0]] == "1") & (df[last_two_columns[1]] == "ENTAILMENT"))
        | ((df[last_two_columns[0]] == 0) & (df[last_two_columns[1]] != "ENTAILMENT"))
        | ((df[last_two_columns[0]] == "0") & (df[last_two_columns[1]] != "ENTAILMENT"))
    ]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_path> <output_csv_path>")
        sys.exit(1)

    input_csv_path: str = sys.argv[1]
    output_csv_path: str = sys.argv[2]

    # Perform the filtering and save the result
    filter_csv_rows(input_csv_path, output_csv_path)

    print(f"Filtered rows saved to {output_csv_path}")
