"""
Script to analyze text columns in a CSV dataset.

Main steps:
- Clean and split text columns into words.
- Remove excluded words and words with digits (optional).
- Apply removal_rules: remove words from target column that appear in source column.
- Count occurrences of each word per column.
- Generate output with per column, having word + word counts.
- Save result as CSV.

Author: Anthony van Tilburg
Date:  July 2, 2025
"""
import pandas as pd
from datetime import datetime
import time as time_module
from typing import List

# --- Start time ---
start_time_run = time_module.time()

# --- LOAD DATASET ---
#df = pd.read_csv("C:/Users/avtnl/Documents//HU/orginele bestanden/US_Accidents_TX_subset.csv")
df = pd.read_csv("C:/Users/avtnl/Documents/HU/Data Visualisation & Visualisation/DAV/data/processed/whatsapp_all_enriched-20251029-162237.csv")

# --- PARAMETERS ---
remove_words_with_digits: bool = False
remove_words_in_first_column_from_second_column: bool = False  # MASTER SWITCH

columns_having_text: List[str] = ['message_cleaned']

# removal_rules: List[str] = [
#     ['Street', 'Description']
# ]

removal_rules: List[str] = []

words_to_exclude: List[str] = ['de', 'het', 'een', 'ik', 'je', 'jij', 'hij', 'zij', 'we', 'wij', 'jullie', 'zij',
                              'mijn', 'jouw', 'zijn', 'haar', 'ons', 'jullie', 'hun', 'dit', 'dat', 'die', 'en', 'of', 'maar', 'want', 'dus', 
                              'ook', 'wel', 'niet', 'al', 'nog', 'toch', 'dan']

# --- SAFETY CHECKS ---
# Lowercase words_to_exclude
words_to_exclude = [w.lower() for w in words_to_exclude]

# Warn if remove_words_in_first_column_from_second_column but < 2 columns
if remove_words_in_first_column_from_second_column and len(columns_having_text) < 2:
    print("WARNING: remove_words_in_first_column_from_second_column is True but there are less than 2 columns_having_text!")

# Warn if any column is not string type
for col in columns_having_text:
    col_dtype = df.dtypes[col]
    if not (pd.api.types.is_string_dtype(df[col])):
        print(f"WARNING: Column '{col}' is of type {col_dtype} — whereas string was expected!")

def validate_removal_rules(removal_rules: List[List[str]], columns_having_text: List[str]) -> List[List[str]]:
    """
    Validate removal_rules.

    Args:
        removal_rules (List[List[str]]): Each rule must be a list of 2 column names [source_col, target_col].
        columns_having_text (List[str]): List of columns being processed.

    Returns:
        List[List[str]]: List of validated rules.
    """
    print(f"Validating removal_rules...")
    valid_rules = []
    for pair in removal_rules:
        if not isinstance(pair, list):
            print(f"WARNING: Invalid entry {pair} in removal_rules (should be a list of 2 columns)")
            continue
        if len(pair) != 2:
            print(f"WARNING: Invalid pair {pair} in removal_rules (should be 2 columns)")
            continue
        col1, col2 = pair
        if col1 not in columns_having_text or col2 not in columns_having_text:
            print(f"WARNING: Column '{col1}' or '{col2}' not found in columns_having_text!")
            continue
        elif columns_having_text.index(col1) >= columns_having_text.index(col2):
            print(f"WARNING: In removal_rules pair {pair}, '{col1}' should appear BEFORE '{col2}' in columns_having_text!")
            continue
        # Valid!
        valid_rules.append(pair)
    print(f"Valid removal_rules: {valid_rules}")
    return valid_rules

# --- Validate removal_rules ---
validated_pairs_in_removal_rules = validate_removal_rules(removal_rules, columns_having_text)

# --- DECIDE: removal_rules active or not ---
if not remove_words_in_first_column_from_second_column:
    print("NOTE: remove_words_in_first_column_from_second_column = False → NO removal will be applied. removal_rules will be IGNORED.")
    removal_rules = []
else:
    if removal_rules:
        print("NOTE: remove_words_in_first_column_from_second_column = True AND removal_rules defined → applying removal_rules.")
    else:
        print("NOTE: remove_words_in_first_column_from_second_column = True BUT no removal_rules defined → using legacy behavior.")

# --- Initialize ---
df_combined_columns = pd.Series(dtype=str)
column_word_sets: dict[str, set[str]] = {}  # key = name of column (example: 'Description') and value is a set of (unique) words; used when applying removal-rules!
df_words_final = pd.Series(dtype=str)

# Initialize column_word_counts for per-column output!
column_word_counts: dict[str, pd.Series] = {}  # key = name of column (example: 'Description') and value is a Series: index = word, value = count


# --- STEP 1: Process each column ---
for col_idx, col in enumerate(columns_having_text):
    print(f"Processing column: {col}")

    # Clean column
    df_single_column = df[col].dropna().str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()  # Specific code to remove punctuation, make lowercase, strip spaces
    df_combined_columns = pd.concat([df_combined_columns, df_single_column], ignore_index=True)

    # Split and explode
    df_words = df_single_column.str.split()  # Generates a list of words in that specific df-cell
    df_words_exploded = df_words.explode().str.strip()  # Generates a new Series having all individual words (one word per row)

    # STEP 3b: Remove excluded words
    df_words_exploded = df_words_exploded[~df_words_exploded.isin(words_to_exclude)]

    # STEP 3c: Remove words with digits (optional)
    if remove_words_with_digits:
        df_words_exploded = df_words_exploded[~df_words_exploded.str.contains(r'\d', regex=True)]  # Specific code to detect words having digits

    # Save unique words for this column
    column_word_sets[col] = set(df_words_exploded.dropna().str.strip().unique())  # Removes None's, strips spaces and consolidates the set to unique values

    # STEP 4: Apply removal_rules if any
    for rule in validated_pairs_in_removal_rules:
        source_col, target_col = rule  # Pairs are split and named source_col (rule[0]) and target_col(rule[1])
        if col == target_col:
            # Prepare for filtering:
            df_words_exploded = df_words_exploded.str.strip()
            # Get source set — safe default empty set if source_col not found
            source_words = column_word_sets.get(source_col, set())

            # FILTERING step:
            df_words_exploded = df_words_exploded[~df_words_exploded.isin(source_words)]

    # After filtering — count words for THIS column
    word_counts_this_col = df_words_exploded.value_counts()  # .value_counts() is a pandas Series method → returns word counts as Series (word → count)

    # Save into column_word_counts
    column_word_counts[col] = word_counts_this_col

# --- STEP 5: Combine per-column counts into DataFrame ---
df_result_as_list: List[pd.DataFrame] = []

for col, word_counts in column_word_counts.items():
    temp_df = word_counts.reset_index()  # to convert Series to df with 2 columns: [word, count]
    temp_df.columns = [col, f'Count_{col}']  # Example for 'Street'-> ['Street', 'Çount_Street']
    df_result_as_list.append(temp_df)

# Combine all columns side by side
df_result = pd.concat(df_result_as_list, axis=1)

# --- STEP 6: Save to CSV ---
DATE_TIME = datetime.now().strftime('%d%b%Y_%H%M')
output_words = f"C:/Users/avtnl/Documents/HU/Data Visualisation & Visualisation/DAV/data/processed/Words_Counts_{DATE_TIME}.csv"
df_result.to_csv(output_words, index=False)

# --- End time ---
end_time_run = time_module.time()
elapsed_seconds = end_time_run - start_time_run
print(f"Total run time: {elapsed_seconds/60:.2f} minutes")
print("End time:", datetime.now().strftime("%H:%M:%S"))
