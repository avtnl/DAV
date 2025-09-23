import re
import pandas as pd

def apply_datetime_fixes_using_re(text: str) -> str:
    """Converts to string and removes trailing fractional seconds like '.123'."""
    if pd.isna(text):
        return None
    # return re.sub(r'\.\d+$', '', str(text).strip())


def apply_basic_re(text: str) -> str:
    """Lowercase and remove all non-word characters except spaces."""
    if pd.isna(text):
        return text
    return re.sub(r'[^\w\s]', '', str(text).lower())


def insert_using_re(text: str) -> str:
    """Insert text beyond text 'rain' or 'drizzle'
       Example: 'Light Rain' becomes 'Light Rain in the Vicinity'
    """
    if pd.isna(text):
        return text
    return re.sub(r'\b(Rain|Drizzle)\b', r'\1 in the Vicinity', text, flags=re.IGNORECASE, count=1)


def apply_split_at_end_number_using_re(text: str) -> str:
    """Split at end of a number.
       Example: 'us183e' becomes 'us183 e' and 'i410loop' becomes 'i410 loop'
    """
    if pd.isna(text):
        return text
    return re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)


def delete_us_space_using_re(text: str) -> str:
    """Delete 'us ' in 'us US-xxx'
       Example: 'us US-183' becomes 'US-183'
    """
    if pd.isna(text):
        return text
    return re.sub(r'\bus\s+(US-\d+)', r'\1', text, flags=re.IGNORECASE)
