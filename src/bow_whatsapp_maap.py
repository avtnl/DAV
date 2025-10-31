#!/usr/bin/env python3
"""
Bag-of-Words keyword search on WhatsApp CSV
-------------------------------------------
- Input: CSV with columns 'whatsapp_group' and 'message_cleaned'
- Filter: only rows where whatsapp_group == 'maap'
- Search: message_cleaned for defined keywords
- Output: matching messages, BoW matrix, keyword totals
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import sys
import os

# -------------------------------------------------
# 1. CONFIG: Your keywords (edit this set!)
# -------------------------------------------------
KEYWORDS = {
    "marjolein", "evelien", "bo", "lars", "mats",
    "max", "red", "bull", "hamilton", "norris", "race", "qualifying", "finish",
    "golf", "handicap", "birdie", "tee", "holes", "par",
    "vakantie", "vliegen", "vliegtuig", "paspoort", "reservering", "zon"
}

# -------------------------------------------------
# 2. Argument parser
# -------------------------------------------------
DEFAULT_CSV = r"C:/Users/avtnl/Documents/HU/Data Visualisation & Visualisation/DAV/data/processed/whatsapp_all_enriched-20251029-162237.csv"
DEFAULT_OUT = r"C:/Users/avtnl/Documents/HU/Data Visualisation & Visualisation/DAV/data/processed/bow.csv"

parser = argparse.ArgumentParser(
    description="BoW search in WhatsApp 'maap' group",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "csv_file",
    nargs="?",                     # makes it optional
    default=DEFAULT_CSV,
    help="Path to the input CSV file"
)
parser.add_argument(
    "--output", "-o",
    default=DEFAULT_OUT,
    help="Path where the matching rows + BoW counts will be saved"
)

args = parser.parse_args()

# Normalise paths (helps with mixed / \ on Windows)
args.csv_file = os.path.abspath(args.csv_file)
args.output   = os.path.abspath(args.output)

# -------------------------------------------------
# 3. Load CSV
# -------------------------------------------------
if not os.path.isfile(args.csv_file):
    print(f"Error: CSV file not found: {args.csv_file}")
    sys.exit(1)

try:
    df = pd.read_csv(args.csv_file)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

required_cols = {'whatsapp_group', 'message_cleaned'}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    print(f"Error: Missing columns in CSV: {missing}")
    sys.exit(1)

# -------------------------------------------------
# 4. Filter: only 'maap' group
# -------------------------------------------------
maap_df = df[df['whatsapp_group'] == 'maap'].copy()

if maap_df.empty:
    print("No messages found in group 'maap'.")
    sys.exit(0)

print(f"Found {len(maap_df)} messages in 'maap' group.")

# -------------------------------------------------
# 5. Extract messages to search
# -------------------------------------------------
messages = maap_df['message_cleaned'].astype(str).tolist()

# -------------------------------------------------
# 6. Build BoW using only your keywords
# -------------------------------------------------
vectorizer = CountVectorizer(vocabulary=KEYWORDS, lowercase=True, binary=False)
bow_matrix = vectorizer.fit_transform(messages)

bow_df = pd.DataFrame(
    bow_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=maap_df.index
)

# -------------------------------------------------
# 7. Find matching messages
# -------------------------------------------------
bow_df['has_match'] = bow_df.sum(axis=1) > 0
matching_indices = bow_df[bow_df['has_match']].index

print(f"\nFound {len(matching_indices)} messages with at least one keyword.\n")

# -------------------------------------------------
# 8. Display matching messages
# -------------------------------------------------
print("="*80)
print("MATCHING MESSAGES (from 'maap' group)")
print("="*80)
for idx in matching_indices:
    original_msg = df.loc[idx, 'message_cleaned']
    print(f"[Row {idx}] {original_msg}")

# -------------------------------------------------
# 9. Summary: total keyword occurrences
# -------------------------------------------------
print("\n" + "="*50)
print("KEYWORD FREQUENCY SUMMARY")
print("="*50)
totals = bow_df.drop(columns='has_match').sum().sort_values(ascending=False)
print(totals[totals > 0])   # only keywords that appeared

# -------------------------------------------------
# 10. Save results
# -------------------------------------------------
result_df = maap_df.loc[matching_indices].copy()
result_df = result_df.join(bow_df.drop(columns='has_match'))

os.makedirs(os.path.dirname(args.output), exist_ok=True)
result_df.to_csv(args.output, index=True)
print(f"\nResults (matches + BoW counts) saved to:\n    {args.output}")

# -------------------------------------------------
# BONUS: Save full BoW matrix (optional)
# -------------------------------------------------
# bow_df.to_csv("bow_matrix_maap.csv")