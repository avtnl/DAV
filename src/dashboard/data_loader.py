# src/dashboard/data_loader.py
import pandas as pd
import streamlit as st
from config import COL
from io import StringIO

@st.cache_data(show_spinner="Loading CSV …")
def load_csv(source):
    """
    `source` can be:
      • a string path (str)
      • an uploaded file object (st.file_uploader → io.BytesIO / io.StringIO)
    """
    if isinstance(source, str):
        df = pd.read_csv(source, low_memory=False)
    else:                                   # uploaded file
        # Streamlit returns BytesIO for binary, but CSV is text → decode
        string_data = StringIO(source.getvalue().decode("utf-8"))
        df = pd.read_csv(string_data, low_memory=False)

    # ----- the rest of the function stays exactly the same -----
    df[COL["timestamp"]] = pd.to_datetime(df[COL["timestamp"]], errors="coerce")
    df[COL["year"]] = df[COL["timestamp"]].dt.year
    df[COL["week"]] = df[COL["timestamp"]].dt.isocalendar().week.astype(int)
    df[COL["day_of_week"]] = df[COL["timestamp"]].dt.day_name()

    def _parse_emojis(val):
        if pd.isna(val):
            return []
        return [e.strip() for e in str(val).split(",") if e.strip()]
    df["emojis_list"] = df[COL["list_emojis"]].apply(_parse_emojis)

    numeric_cols = [
        COL["length_chat"], COL["num_words"], COL["num_emojis"],
        COL["num_punct"], COL["num_capitals"], COL["capital_ratio"],
        COL["num_numbers"]
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    bool_map = {True: True, False: False, "True": True, "False": False}
    for c in [COL["has_emoji"], COL["has_attachment"], COL["has_punct"], COL["has_capitals"]]:
        df[c] = df[c].map(bool_map).fillna(False)

    return df