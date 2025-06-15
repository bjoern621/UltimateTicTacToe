# balance_data.py
import pandas as pd

# --- Configuration ---
INPUT_CSV_FILE = "boards_dataset_complete_filtered.csv"
OUTPUT_CSV_FILE = "boards_dataset_complete_filtered_balanced.csv"
TARGET_COLUMN = "winner"  # IMPORTANT: Change this to the name of your column that stores 'X', 'O', 'DRAW'
LIMIT_PER_CATEGORY = 9923


def balance_tic_tac_toe_data(input_file, output_file, target_col, limit):
    """
    Balances the dataset so that each category in the target_col
    has at most 'limit' entries.

    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the balanced CSV file.
        target_col (str): The name of the column containing game outcomes (e.g., 'X', 'O', 'DRAW').
        limit (int): The maximum number of entries per category.
    """
    try:
        df = pd.read_csv(input_file, sep=";")
        print(f"Successfully loaded '{input_file}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV '{input_file}': {e}")
        return

    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in the CSV.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    print(f"\nOriginal value counts for column '{target_col}':")
    print(df[target_col].value_counts())
    print("-" * 30)

    # This is the core logic:
    # 1. Group the DataFrame by the target_col.
    # 2. For each group, take the first 'limit' rows using .head(limit).
    #    If a group has fewer than 'limit' rows, all its rows are taken.
    # group_keys=False prevents the group key from being added as an index level,
    # which simplifies the resulting DataFrame.
    balanced_df = df.groupby(target_col, group_keys=False).apply(
        lambda x: x.head(limit)
    )

    # If you wanted a random sample instead of the first N:
    # def sample_or_all(x, n):
    #     if len(x) > n:
    #         return x.sample(n=n, random_state=42) # random_state for reproducibility
    #     return x
    # balanced_df = df.groupby(target_col, group_keys=False).apply(lambda x: sample_or_all(x, limit))

    print(f"\nBalanced dataset value counts for column '{target_col}':")
    print(balanced_df[target_col].value_counts())
    print("-" * 30)

    try:
        balanced_df.to_csv(output_file, index=False, sep=";")
        print(f"Balanced dataset saved to '{output_file}'. Shape: {balanced_df.shape}")
    except Exception as e:
        print(f"Error writing balanced data to '{output_file}': {e}")


if __name__ == "__main__":
    # --- Create a dummy boards_dataset.csv for demonstration if it doesn't exist ---

    pd.read_csv(INPUT_CSV_FILE, sep=";")

    balance_tic_tac_toe_data(
        INPUT_CSV_FILE, OUTPUT_CSV_FILE, TARGET_COLUMN, LIMIT_PER_CATEGORY
    )
