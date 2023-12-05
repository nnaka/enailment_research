import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_stacked_bar_chart(csv_path: str, output_path: str) -> None:
    """
    Generate a stacked bar chart from a CSV file and save it as a PDF.

    Args:
        csv_path (str): Path to the input CSV file.
        output_path (str): Path for the output PDF file.
    """
    # Read CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Transpose the DataFrame to use rows as the magnitude of the stacks
    df_transposed = df.set_index(df.columns[0]).transpose()

    # Plot a stacked bar chart
    ax = df_transposed.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Add labels and title
    ax.set_xlabel("Data Source")
    ax.set_ylabel("Counts")
    ax.set_title("Comparison of Entailment Types Across Data Sources")

    # Add annotations for each height value
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        # Make sure to omit "0.0" at the bottom axis
        if height > 0:
            x, y = p.get_xy() 
            ax.annotate(f'{height}', (x + width/2, y + height/2), ha='center', va='center')

    # Add legend within the bounds of the chart
    ax.legend(title="Entailment Types", bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.legend(title="Entailment Types", loc='best')


    # Save the plot as a PDF file
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_path> <output_pdf_path>")
        sys.exit(1)

    input_csv_path: str = sys.argv[1]
    output_pdf_path: str = sys.argv[2]

    # Generate the stacked bar chart and save as a PDF
    generate_stacked_bar_chart(input_csv_path, output_pdf_path)

    print(f"Stacked bar chart saved to {output_pdf_path}")

