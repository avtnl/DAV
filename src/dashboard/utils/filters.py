# utils/filters.py
import pandas as pd
import streamlit as st
from config import COL


def sidebar_filters(df: pd.DataFrame):
    # ------------------------------------------------------------------
    # 1. Year slider
    # ------------------------------------------------------------------
    min_year = int(df[COL["year"]].min())
    max_year = int(df[COL["year"]].max())
    year_range = st.sidebar.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
    )

    # ------------------------------------------------------------------
    # 2. Message-type radio
    # ------------------------------------------------------------------
    msg_type = st.sidebar.radio(
        "Message filter",
        [
            "All Messages",
            "Has Emoji",
            "Has Attachment",
            "Has Punctuation",
            "Has Capitals",
        ],
        index=0,
    )

    # ------------------------------------------------------------------
    # 3. Length slider
    # ------------------------------------------------------------------
    max_len = int(df[COL["length_chat"]].max())
    len_range = st.sidebar.slider(
        "Message length (characters)",
        min_value=0,
        max_value=max_len,
        value=(0, max_len),
        step=1,
    )

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------
    fdf = df.copy()
    fdf = fdf[(fdf[COL["year"]] >= year_range[0]) & (fdf[COL["year"]] <= year_range[1])]

    if msg_type == "Has Emoji":
        fdf = fdf[fdf[COL["has_emoji"]]]
    elif msg_type == "Has Attachment":
        fdf = fdf[fdf[COL["has_attachment"]]]
    elif msg_type == "Has Punctuation":
        fdf = fdf[fdf[COL["has_punct"]]]
    elif msg_type == "Has Capitals":
        fdf = fdf[fdf[COL["has_capitals"]]]

    fdf = fdf[fdf[COL["length_chat"]].between(len_range[0], len_range[1])]

    return {
        "df": fdf,
        "year_range": year_range,
        "msg_type": msg_type,
        "len_range": len_range,
    }
