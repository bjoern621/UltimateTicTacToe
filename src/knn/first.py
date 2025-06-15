import pandas as pd
import os


def filter_first_entries(input_file, output_file=None):
    """
    Filter a dataset to keep only the first entry from each category 'i'.

    Parameters:
    -----------
    input_file : str
        Path to the input data file (CSV, Excel, etc.)
    output_file : str, optional
        Path to save the filtered data. If None, will generate based on input filename.

    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame
    """
    # Determine file extension for reading
    _, ext = os.path.splitext(input_file)

    # Read the data based on file extension
    if ext.lower() == ".csv":
        df = pd.read_csv(input_file, sep=";")
    elif ext.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_file, sep=";")
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Please use .csv or .xlsx/.xls files."
        )

    # Identify the column representing the 'i' category
    # This assumes your category column is actually named 'i'
    # If it has a different name, replace 'i' with the actual column name
    category_col = "i"

    # Keep only the first occurrence of each category 'i'
    filtered_df = df.drop_duplicates(subset=[category_col], keep="first")

    # Generate output filename if not provided
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"

    # Save the filtered data
    if ext.lower() == ".csv":
        filtered_df.to_csv(output_file, index=False)
    else:
        filtered_df.to_excel(output_file, index=False)

    print(f"Original data has {len(df)} rows.")
    print(f"Filtered data has {len(filtered_df)} rows.")
    print(f"Filtered data saved to {output_file}")

    return filtered_df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        filter_first_entries(input_file, output_file)
    else:
        print("Usage: python filter_data.py input_file [output_file]")
        print("Example: python filter_data.py data.csv filtered_data.csv")
