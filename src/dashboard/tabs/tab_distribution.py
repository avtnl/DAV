# src/dashboard/tabs/tab_distribution.py
from ast import literal_eval

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from config import COL


def render_distribution_tab(df_filtered) -> None:
    st.header("Distribution – Emoji Usage")

    # ------------------------------------------------------------------
    # 1. Mode: Overview / Test Probability
    # ------------------------------------------------------------------
    mode = st.radio("Mode", ["Overview", "Test Probability"], horizontal=True, key="dist_mode")

    # ------------------------------------------------------------------
    # 2. Parse emojis (robust)
    # ------------------------------------------------------------------
    def safe_parse_emojis(x):
        if pd.isna(x) or not x:
            return []
        s = str(x).strip()
        if s in ("[]", "''", '""', "nan"):
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                return literal_eval(s)
            except:
                pass
        return [e.strip().strip("'\"") for e in s.split(",") if e.strip()]

    df_filtered["emojis_parsed"] = df_filtered[COL["list_emojis"]].apply(safe_parse_emojis)
    df_filtered["has_emoji_flag"] = df_filtered["emojis_parsed"].apply(len) > 0
    df_with_emoji = df_filtered[df_filtered["has_emoji_flag"]].copy()

    if df_with_emoji.empty:
        st.error("No messages with emojis found.")
        return

    # ------------------------------------------------------------------
    # OVERVIEW MODE
    # ------------------------------------------------------------------
    if mode == "Overview":
        render_overview(df_filtered, df_with_emoji)

    # ------------------------------------------------------------------
    # TEST PROBABILITY MODE
    # ------------------------------------------------------------------
    else:
        render_probability_game(df_filtered, df_with_emoji)


# ======================================================================
# OVERVIEW
# ======================================================================
def render_overview(df_filtered, df_with_emoji) -> None:
    # Group selector
    group_options = ["all whatsapp_groups", *sorted(df_filtered[COL["group"]].unique().tolist())]
    selected_group = st.selectbox("Select Group", group_options, key="overview_group")

    if selected_group == "all whatsapp_groups":
        df_group = df_with_emoji.copy()
        title_suffix = "All Groups"
    else:
        df_group = df_with_emoji[df_with_emoji[COL["group"]] == selected_group].copy()
        title_suffix = selected_group

    # Explode + count
    df_group["emojis_unique"] = df_group["emojis_parsed"].apply(lambda x: list(set(x)))
    exploded = df_group.explode("emojis_unique")
    emoji_counts = exploded["emojis_unique"].value_counts().reset_index()
    emoji_counts.columns = ["emoji", "count"]

    total_messages = len(df_group)

    # Max slider (controls how many emojis to show)
    max_emojis = st.slider(
        "Max # of Emojis to Show",
        min_value=1,
        max_value=len(emoji_counts),
        value=min(60, len(emoji_counts)),
        key="max_emojis_slider",
    )

    top_n = emoji_counts.head(max_emojis).copy()
    top_n["likelihood"] = top_n["count"] / total_messages
    top_n["cumulative"] = top_n["likelihood"].cumsum()

    # Plot
    fig = go.Figure()

    # Bars: likelihood
    fig.add_trace(
        go.Bar(
            x=top_n["emoji"],
            y=top_n["likelihood"],
            name="Likelihood",
            marker_color="lightblue",
            text=top_n["emoji"],
            textposition="outside" if len(top_n) <= 26 else "none",
        )
    )

    # Orange line: cumulative
    fig.add_trace(
        go.Scatter(
            x=top_n["emoji"],
            y=top_n["cumulative"],
            mode="lines+markers",
            name="Cumulative",
            line={"color": "orange", "width": 3},
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=f"Emoji Likelihood – {title_suffix}<br><sub>Top {len(top_n)} of {len(emoji_counts)} emojis</sub>",
        xaxis_title="Emoji",
        yaxis={"title": "Likelihood", "tickformat": ".0%", "range": [0, 1]},
        yaxis2={
            "title": "Cumulative",
            "tickformat": ".0%",
            "range": [0, 1],
            "overlaying": "y",
            "side": "right",
        },
        height=600,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Final counts below
    with st.expander("Final Emoji Counts", expanded=False):
        st.dataframe(
            top_n[["emoji", "count", "likelihood", "cumulative"]].style.format(
                {"likelihood": "{:.1%}", "cumulative": "{:.1%}"}
            )
        )


# ======================================================================
# TEST PROBABILITY GAME
# ======================================================================
def render_probability_game(df_filtered, df_with_emoji) -> None:
    st.subheader("Test Emoji Probability")

    col1, col2 = st.columns(2)

    # Emoji selector
    all_emojis = sorted({e for lst in df_with_emoji["emojis_parsed"] for e in lst})
    selected_emoji = col1.selectbox("Select Emoji", all_emojis, key="test_emoji")

    # Group selector
    group_options = ["all whatsapp_groups", *sorted(df_filtered[COL["group"]].unique().tolist())]
    selected_group = col2.selectbox("WhatsApp Group", group_options, key="test_group")

    # Author selector + sync
    if selected_group == "all whatsapp_groups":
        author_options = ["all authors", *sorted(df_filtered[COL["author"]].unique().tolist())]
    else:
        authors_in_group = df_filtered[df_filtered[COL["group"]] == selected_group][
            COL["author"]
        ].unique()
        author_options = ["all authors", *sorted(authors_in_group)]

    selected_author = col1.selectbox("Author", author_options, key="test_author")

    # Sync: if author not in group → switch group
    if selected_author != "all authors" and selected_group != "all whatsapp_groups":
        author_group = df_filtered[df_filtered[COL["author"]] == selected_author][
            COL["group"]
        ].unique()
        if selected_group not in author_group and len(author_group) > 0:
            new_group = author_group[0]
            st.warning(
                f"Author '{selected_author}' not in '{selected_group}'. Switching to '{new_group}'."
            )
            selected_group = new_group

    # Filter data
    df_test = df_with_emoji.copy()
    if selected_group != "all whatsapp_groups":
        df_test = df_test[df_test[COL["group"]] == selected_group]
    if selected_author != "all authors":
        df_test = df_test[df_test[COL["author"]] == selected_author]

    total_messages = len(df_test)
    emoji_occurrences = df_test["emojis_parsed"].apply(lambda x: selected_emoji in x).sum()
    probability = emoji_occurrences / total_messages if total_messages > 0 else 0

    st.metric(
        "Probability", f"{probability:.1%}", help=f"{emoji_occurrences} / {total_messages} messages"
    )

    # Sample size + tests
    col3, col4 = st.columns(2)
    max_sample = min(total_messages, 1000)
    sample_size = col3.slider("Sample Size", 1, max_sample, min(100, max_sample), key="sample_size")
    num_tests = col4.slider("Number of Tests", 1, 100, 10, key="num_tests")

    if st.button("Run Test"):
        results = []
        for _ in range(num_tests):
            sample = df_test.sample(n=sample_size, replace=False)
            found = sample["emojis_parsed"].apply(lambda x: selected_emoji in x).sum()
            results.append(found)

        expected = sample_size * probability
        actual_avg = np.mean(results)

        st.write(f"**Expected per sample:** {expected:.2f}")
        st.write(f"**Actually found (avg):** {actual_avg:.2f}")

        diff = actual_avg - expected
        if diff > 0.5 * expected:
            st.success("**Exceeds expectation!** 🎉 You're on a hot streak!")
        elif abs(diff) <= 0.3 * expected:
            st.info("**In line with expectation.** 📊 Solid science.")
        else:
            st.warning("**Below expectation.** 😔")
            if probability < 0.05:
                st.write("→ You selected a **rare emoji**.")
            if sample_size < 50:
                st.write("→ Try **increasing sample size**.")
            if num_tests < 20:
                st.write("→ Try **more tests** for better stats.")
