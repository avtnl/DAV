# src/dashboard/utils/style_analyzer.py
import numpy as np
import pandas as pd
import streamlit as st
from config import COL
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@st.cache_data(show_spinner="Computing style fingerprint…")
def compute_style_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [COL["author"], COL["year"], "whatsapp_group_temp"]

    agg_dict = {
        "msg_count": (COL["message_clean"], "count"),
        "words_total": (COL["num_words"], "sum"),
        "emojis_total": (COL["num_emojis"], "sum"),
        "punct_total": (COL["num_punct"], "sum"),
        "caps_total": (COL["num_capitals"], "sum"),
        "mean_chat_len": (COL["length_chat"], "mean"),
    }

    if "avg_word_length" in df.columns:
        agg_dict["mean_word_len"] = ("avg_word_length", "mean")

    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()

    mc = agg["msg_count"].replace(0, np.nan)
    agg["words_per_msg"] = agg["words_total"] / mc
    agg["emoji_per_msg"] = agg["emojis_total"] / mc
    agg["punct_per_msg"] = agg["punct_total"] / mc
    agg["caps_per_word"] = agg["caps_total"] / agg["words_total"].replace(0, np.nan)

    agg["pct_ends_emoji"] = 0.0
    agg["pct_ends_punct"] = 0.0
    agg["pct_has_question"] = 0.0

    if COL.get("ends_with_emoji") in df.columns:
        agg["pct_ends_emoji"] = df.groupby(group_cols)[COL["ends_with_emoji"]].mean().values
    if COL.get("ends_with_punctuation") in df.columns:
        agg["pct_ends_punct"] = df.groupby(group_cols)[COL["ends_with_punctuation"]].mean().values
    if COL.get("has_question_mark") in df.columns:
        agg["pct_has_question"] = df.groupby(group_cols)[COL["has_question_mark"]].mean().values

    style_cols = [
        "words_per_msg",
        "emoji_per_msg",
        "punct_per_msg",
        "caps_per_word",
        "mean_chat_len",
    ]
    if "mean_word_len" in agg.columns:
        style_cols.append("mean_word_len")
    style_cols += ["pct_ends_emoji", "pct_ends_punct", "pct_has_question"]

    X = agg[style_cols].fillna(0).values
    if X.shape[0] == 0:
        return pd.DataFrame(columns=[*group_cols, "tsne_x", "tsne_y", "msg_count"])

    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(5, X.shape[1])
    X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)

    perplexity = min(30, len(X) - 1) if len(X) > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=300)
    coords = tsne.fit_transform(X_pca)

    agg["tsne_x"] = coords[:, 0]
    agg["tsne_y"] = coords[:, 1]
    return agg


@st.cache_data(show_spinner="Computing message fingerprint…")
def compute_message_fingerprint(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [COL["author"], COL["year"], "whatsapp_group_temp"]

    grouped = (
        df.groupby(group_cols)["message_cleaned"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )
    grouped["message_cleaned"] = grouped["message_cleaned"].str.strip()
    grouped = grouped[grouped["message_cleaned"] != ""]

    if grouped.empty:
        return pd.DataFrame(columns=[*group_cols, "tsne_x", "tsne_y", "msg_count"])

    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", lowercase=True)
    X = vectorizer.fit_transform(grouped["message_cleaned"]).toarray()

    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(5, X.shape[1])
    X_pca = PCA(n_components=n_comp, random_state=42).fit_transform(X_scaled)

    perplexity = min(30, len(X) - 1) if len(X) > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=300)
    coords = tsne.fit_transform(X_pca)

    result = grouped.copy()
    result["tsne_x"] = coords[:, 0]
    result["tsne_y"] = coords[:, 1]
    result["msg_count"] = (
        df.groupby(group_cols)
        .size()
        .reindex(result.set_index(group_cols).index, fill_value=0)
        .values
    )
    return result
