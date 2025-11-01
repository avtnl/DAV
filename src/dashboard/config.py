# config.py

# ----------------------------------------------------------------------
# Column names
# ----------------------------------------------------------------------
COL = {
    "timestamp": "timestamp",
    "author": "author",
    "group": "whatsapp_group",
    "group_temp": "whatsapp_group_temp",
    "message": "message",
    "message_clean": "message_cleaned",
    "year": "year",
    "month": "month",
    "week": "week",
    "day_of_week": "day_of_week",
    "length_chat": "length_chat",
    "num_words": "number_of_words",
    "num_emojis": "number_of_emojis",
    "has_emoji": "has_emoji",
    "list_emojis": "list_of_all_emojis",
    "has_attachment": "has_attachment",
    "num_punct": "number_of_punctuations",
    "has_punct": "has_punctuation",
    "num_capitals": "number_of_capitals",
    "has_capitals": "has_capitals",
    "capital_ratio": "capitalized_words_ratio",
    "num_numbers": "number_of_numbers",
    "has_numbers": "has_number_characters",
}

# ----------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------
GROUP_COLORS: dict[str, str] = {
    "maap": "#1f77b4",
    "dac": "#2ca02c",
    "golfmaten": "#ff7f0e",
    "tillies": "#7f7f7f",
    "AvT": "#aec7e8",
}


# AvT = blue, others = light gray
def author_color(author: str) -> str:
    return "#1f77b4" if author == "AvT" else "#f0f0f0"  # light gray
