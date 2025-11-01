print("\nSTARTING SCRIPT...")

import time as time_module
from datetime import datetime

import pandas as pd


def build_structures_part1(
    df_structures: pd.DataFrame,
    df_words: pd.Series,
    words_to_include: list[str],
    phrases_to_exclude: list[str],
    prefixes_to_exclude: list[str],
) -> pd.DataFrame:
    """
    Adds one column per word in words_to_include and updates 'Structures_Summary'.
    For each row in df_words:
    - Skips excluded phrases (prefix + next word).
    - Adds only df_words starting from the first occurrence of matching words in words_to_include.
    - Structures_Summary is ordered by the actual order of appearance in the sentence.

    Args:
        df_structures: Dataframe to update (must contain 'Structures_Summary' column).
        df_words: Series of lists of words (tokenized text).
        words_to_include (listed by user): Words to include (target words to match).
        phrases_to_exclude (listed by user): Phrases to exclude (prefix + next word).
        prefixes_to_exclude: Prefixes for excluded phrases.

    Returns:
        pd.DataFrame: Updated df_structures with one column per word in words_to_include and updated 'Structures_Summary'
    """
    print("\n--- STEP 1: Building Part 1 ---")

    # --- Validate inputs ---
    for word in words_to_include:
        if " " in word:
            print(f"WARNING: word_to_include '{word}' contains spaces — must be a single word!")

    for phrase in phrases_to_exclude:
        if phrase.count(" ") != 1:
            print(
                f"WARNING: phrase_to_exclude '{phrase}' must contain exactly 2 words (one space)!"
            )

    for idx, word_list in df_words.items():
        if isinstance(word_list, float):
            df_structures.at[idx, "Structures_Summary"] = []
            for word in words_to_include:
                df_structures.at[idx, word] = ""
            continue

        # List of (position, word)
        positions = []

        # We will store the result_string for each word
        result_strings = {}

        # Pre-scan entire sentence once:
        idx2 = 0
        while idx2 < len(word_list):
            current_word = word_list[idx2]

            # Check exclude combo
            skip_phrase = False
            if current_word in prefixes_to_exclude:
                if idx2 + 1 < len(word_list):
                    next_word = word_list[idx2 + 1]
                    combo = f"{current_word} {next_word}"
                    if combo in phrases_to_exclude:
                        # Skip this phrase
                        idx2 += 2
                        skip_phrase = True

            if not skip_phrase:
                # Check if current word is one of the words_to_include
                if current_word in words_to_include:
                    if current_word not in result_strings:
                        # First occurrence: store position + build result string
                        positions.append((idx2, current_word))
                        result_string = " ".join(word_list[idx2:])
                        result_strings[current_word] = result_string

                idx2 += 1

        # Now sort positions by position (order of appearance)
        positions.sort()
        df_structures.at[idx, "Structure_Positions"] = [pos for pos, word in positions]

        # Build Structures_Summary in appearance order
        structures_summary_row = [word for pos, word in positions]

        # Save columns
        for word in words_to_include:
            if word in result_strings:
                df_structures.at[idx, word] = result_strings[word]
            else:
                df_structures.at[idx, word] = ""

        # Save Structures_Summary
        df_structures.at[idx, "Structures_Summary"] = structures_summary_row

    return df_structures


def add_no_matches(df_structures: pd.DataFrame, df_base: pd.Series) -> pd.DataFrame:
    """
    Adds a 'No_Matches' column to df_structures.
    For each row:
    - If 'Structures_Summary' is empty ([]) or None, copies the corresponding value from df_base.
    - Otherwise, sets 'No_Matches' to None.

    Args:
        df_structures (pd.DataFrame): The dataframe to update (must have 'Structures_Summary' column).
        df_base (pd.Series): The reference Series (cleaned text).

    Returns:
        pd.DataFrame: Updated df_structures with 'No_Matches' column.
    """
    print("\n--- STEP 2: Adding 'No_Matches' ---")

    # Create the column (if not exists yet)
    df_structures["No_Matches"] = None

    # Process each row
    for idx in df_structures.index:
        structures_summary = df_structures.at[idx, "Structures_Summary"]

        if structures_summary is None or structures_summary == []:
            df_structures.at[idx, "No_Matches"] = df_base.at[idx]
        else:
            df_structures.at[idx, "No_Matches"] = None

    return df_structures


def build_structures_part2(
    df_structures: pd.DataFrame, words_to_include: list[str]
) -> pd.DataFrame:
    """
    Adds one column per word in words_to_include. Every added column contains all words from the corresponding column added in build_structures_part1,
    skipping the words starting from the next word in words_to_include (using string length cuts).

    The processing order is now based on Structure_Positions (order of appearance in sentence).

    Args:
        df_structures: Dataframe to update (must have Structures_Summary and Structure_Positions columns).
        words_to_include: List of structure words.

    Returns:
        pd.DataFrame: Updated df_structures with one extra column per word in words_to_include
    """
    print("\n--- STEP 3: Building Part 2 ---")

    cleaned_columns = [f"{word}_cleaned" for word in words_to_include]
    df_cleaned = pd.DataFrame(index=df_structures.index, columns=cleaned_columns)

    for idx in df_structures.index:
        # Initialize memory
        memorized_string_length_1 = 0
        memorized_string_length_2 = 0

        cleaned_values = []

        structures_summary = df_structures.at[idx, "Structures_Summary"]
        structure_positions = df_structures.at[idx, "Structure_Positions"]

        if not structures_summary or not isinstance(structures_summary, list):
            # The whole row is empty → store None for the reverse columns → skip row
            df_cleaned.loc[idx] = None
            continue

        # Build the actual order of words based on positions
        # This guarantees correct processing order
        positions_sorted = sorted(zip(structure_positions, structures_summary, strict=False))
        words_in_columns = [word for pos, word in positions_sorted][::-1]  # reversed

        # Check if row is empty (first word column is None) — EARLY EXIT!
        first_word = words_in_columns[0]
        first_full_string = df_structures.at[idx, first_word]

        if first_full_string is None:
            df_cleaned.loc[idx] = None
            continue

        # Now process normally — using string length cuts
        for word in words_in_columns:
            full_string = df_structures.at[idx, word]

            if full_string is None:
                cleaned_values.append("")
                continue

            # Rest of normal processing here:
            memorized_string_length_2 = len(full_string)

            if memorized_string_length_1 > 0:
                cut_length = memorized_string_length_2 - memorized_string_length_1
                current_substring = full_string[:cut_length].rstrip()
            else:
                current_substring = full_string.rstrip()

            first_word_length = len(word) + 1
            if len(current_substring) > first_word_length:
                result_string = current_substring[first_word_length:].lstrip()
            else:
                result_string = ""

            memorized_string_length_1 = memorized_string_length_2

            cleaned_values.append(result_string)

        # Reverse the cleaned_values back → now it is in sentence order!
        cleaned_values = cleaned_values[::-1]

        # 🚀 Now store correctly — this is the KEY fix:
        # structures_summary gives us the correct sentence order!
        for i, word in enumerate(structures_summary):
            col_name = f"{word}_cleaned"
            df_cleaned.at[idx, col_name] = cleaned_values[i]

        # For structure words not present → set empty string
        for word in words_to_include:
            col_name = f"{word}_cleaned"
            if word not in structures_summary:
                df_cleaned.at[idx, col_name] = ""

    # Finally, copy columns back to df_structures
    for col_name in cleaned_columns:
        df_structures[col_name] = df_cleaned[col_name]

    return df_structures


def add_primary_and_secondary_structures(df_structure: "pd.DataFrame") -> "pd.DataFrame":
    """
    Adds and fills 'Primary_Structure' and 'Secondary_Structure' columns in the given dataframe.

    The logic is based on analyzing the 'Structures_Summary' column, which should contain lists of structure keywords.

    Rules:
    1. If 'on' is present → Primary = 'on', Secondary = rest (with special handling for 'from' and 'to')
    2. If no 'on', but 'at' is present → Primary = 'at', Secondary = rest
    3. If 'from' and 'to' are present (and no 'on' or 'at') → Primary = 'from, to', Secondary = rest
    4. If none of the above, then Primary = first element, Secondary = rest

    Parameters:
        df_structure (pd.DataFrame): DataFrame containing 'Structures_Summary' column (list of keywords)

    Returns:
        pd.DataFrame: DataFrame with 'Primary_Structure' and 'Secondary_Structure' columns filled in
    """
    print("\n--- STEP 4: Building Primary & Secondary Structures ---")
    # Add empty columns
    df_structures["Primary_Structure"] = ""
    df_structures["Secondary_Structure"] = ""

    for index, row in df_structure.iterrows():
        structure = [x.lower() for x in row["Structures_Summary"]]

        # Handle empty case
        if not structure:
            df_structure.at[index, "Primary_Structure"] = ""
            df_structure.at[index, "Secondary_Structure"] = ""
            continue

        # Rule 1: If 'on' present — 'on' is primary
        if "on" in structure:
            primary = "on"

            # Special case: if 'from' and 'to' present, group them together
            if "from" in structure and "to" in structure:
                secondary_parts = ["from", "to"]
                other_terms = [x for x in structure if x not in ("on", "from", "to")]
                secondary_parts.extend(other_terms)
                secondary = ", ".join(secondary_parts)
            else:
                secondary = ", ".join([x for x in structure if x != "on"])

            df_structure.at[index, "Primary_Structure"] = primary
            df_structure.at[index, "Secondary_Structure"] = secondary
            continue

        # Rule 2: If no 'on', but 'at' present — 'at' is primary
        if "at" in structure:
            primary = "at"
            secondary = ", ".join([x for x in structure if x != "at"])

            df_structure.at[index, "Primary_Structure"] = primary
            df_structure.at[index, "Secondary_Structure"] = secondary
            continue

        # Rule 3: If 'from' and 'to' present (and no 'on'/'at') — group them as primary
        if "from" in structure and "to" in structure:
            primary = "from, to"
            other_terms = [x for x in structure if x not in ("from", "to")]
            secondary = ", ".join(other_terms)

            df_structure.at[index, "Primary_Structure"] = primary
            df_structure.at[index, "Secondary_Structure"] = secondary
            continue

        # Rule 4: Fallback — first keyword is primary, rest is secondary
        primary = structure[0]
        secondary = ", ".join(structure[1:])

        df_structure.at[index, "Primary_Structure"] = primary
        df_structure.at[index, "Secondary_Structure"] = secondary

    return df_structure


def add_primary_and_secondary_locations(
    df_structures: "pd.DataFrame", words_to_include: list
) -> "pd.DataFrame":
    """
    Adds 'Primary_Location' and 'Secondary_Location' columns to df_structures.

    Logic:
    - Primary_Location:
        if Primary_Structure == 'on' → use df['on_cleaned'] (via mapping)
    - Secondary_Location:
        if Secondary_Structure == 'from, to' → combine df['from_cleaned'] + df['to_cleaned']
        if Secondary_Structure == 'x, y' → combine columns for 'x' and 'y'
        if Secondary_Structure == 'x' → take column for 'x'

    Args:
        df_structures (pd.DataFrame): dataframe with Primary_Structure, Secondary_Structure and *_cleaned columns
        words_to_include (list): list of structure keywords (example: ['on', 'at', ...])

    Returns:
        pd.DataFrame: dataframe with Primary_Location and Secondary_Location columns added
    """
    print("\n--- STEP 5: Building Primary & Secondary Locations ---")

    # Add empty columns
    df_structures["Primary_Location"] = ""
    df_structures["Secondary_Location"] = ""

    # Build mapping from word → column name
    words_to_column_mapping = {word: f"{word}_cleaned" for word in words_to_include}

    # Iterate over dataframe
    for index, row in df_structures.iterrows():
        ### Primary Location
        primary_structure = row["Primary_Structure"]

        primary_col = words_to_column_mapping.get(primary_structure)
        ##########
        # Fallback: use 'from_cleaned' if structure is 'from, to'
        if primary_structure == "from, to":
            primary_col = words_to_column_mapping.get("from")
        ##########

        if primary_col and primary_col in df_structures.columns:
            df_structures.at[index, "Primary_Location"] = row[primary_col]
        else:
            df_structures.at[index, "Primary_Location"] = ""

        ### Secondary Location
        secondary_structure = row["Secondary_Structure"]

        if not secondary_structure:
            df_structures.at[index, "Secondary_Location"] = ""
            continue

        # Split on comma if needed
        secondary_parts = [part.strip() for part in secondary_structure.split(",")]

        secondary_values = []

        for part in secondary_parts:
            secondary_col = words_to_column_mapping.get(part)
            if secondary_col and secondary_col in df_structures.columns:
                secondary_values.append(row[secondary_col])

        # Combine all secondary values
        df_structures.at[index, "Secondary_Location"] = " ".join(secondary_values)

    return df_structures


# MAIN
# Start time (track total script run time)
start_time_run = time_module.time()

# Load the full TX-dataset
df = pd.read_csv("C:/Users/avtnl/Documents//HU/Orginele bestanden/US_Accidents_TX_subset.csv")
# df = pd.read_csv("C:/Users/avtnl/Documents//HU/Bestanden (output code)/US_Accidents_TX_subset_amended_16Jun2025_0050.csv")

# Parameters
column_name = "Description"
words_to_include = [
    "on",
    "at",
    "near",
    "next",
    "between",
    "and",
    "from",
    "to",
    "after",
    "before",
    "with",
    "in",
    "around",
]
words_to_exclude = ["on exit", "due to", "at exit"]
generate_primary_and_secondary_locations: bool = True
# generate_primary_and_secondary_locations controls whether to generate 'Primary_Structure', 'Secondary_Structure', 'Primary_Location' and 'Secondary_Location'
# If set to False, these columns will not be created

# Pre-processing
df_base = df[column_name].astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
df_words = df_base.str.split()
prefixes_to_exclude = sorted({phrase.split()[0] for phrase in words_to_exclude})

# Make sure we preserve alignment by using df.index
df_structures = pd.DataFrame(index=df.index, columns=["Structures_Summary", "Structure_Positions"])
df_structures["ID"] = df["ID"].values  # Assign the correct ID values by index

# Build structure
df_structures = build_structures_part1(
    df_structures, df_words, words_to_include, words_to_exclude, prefixes_to_exclude
)

# Add No_Matches
add_no_matches(df_structures, df_words)

# Complete structure
df_structures = build_structures_part2(df_structures, words_to_include)

# Generate frequency (counts) per unique structures (including no structures at all)
count_of_structures = df_structures["Structures_Summary"].astype(str).value_counts()

# Add_primary_and_secondary 'Structures' and 'Locations'
if generate_primary_and_secondary_locations:
    # Add columns at the end of df_structures
    df_structures = add_primary_and_secondary_structures(df_structures)

    # Add columns at the end of df_structures
    df_structures = add_primary_and_secondary_locations(df_structures, words_to_include)

# Save output
df_frequency = count_of_structures.reset_index()
df_frequency.columns = ["Structures", "Count"]

DATE_TIME = datetime.now().strftime("%d%b%Y_%H%M")
frequency_file = (
    f"C:/Users/avtnl/Documents/HU/Bestanden (output code)/Structures_Frequency_{DATE_TIME}.csv"
)
df_frequency.to_csv(frequency_file, index=False)

details_file = (
    f"C:/Users/avtnl/Documents/HU/Bestanden (output code)/Structures_All_Details_{DATE_TIME}.csv"
)
df_structures.to_csv(details_file, index=False)

# Extract primary and secondary 'Structures' and 'Locations' only as separate csv-file
if generate_primary_and_secondary_locations:
    df_subset = df_structures[
        ["ID", "Primary_Structure", "Secondary_Structure", "Primary_Location", "Secondary_Location"]
    ]
    details_file = f"C:/Users/avtnl/Documents/HU/Bestanden (output code)/Structures_Extracted_Details_{DATE_TIME}.csv"
    df_subset.to_csv(details_file, index=False)

# End time (track total script run time)
end_time_run = time_module.time()
elapsed_seconds = end_time_run - start_time_run
print("End time:", datetime.now().strftime("%H:%M:%S"))
# Convert elapsed seconds to HH:MM:SS
hours = int(elapsed_seconds // 3600)
minutes = int((elapsed_seconds % 3600) // 60)
seconds = int(elapsed_seconds % 60)
print(f"Total run time: {hours:02}:{minutes:02}:{seconds:02}")
