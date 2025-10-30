# src/dashboard/tabs/tab_distribution.py
import streamlit as st
import plotly.express as px
import pandas as pd
from config import COL
from ast import literal_eval

def render_distribution_tab(df_filtered):
    st.header("Distribution – Emoji Usage")

    # ------------------------------------------------------------------
    # 1. Fix column name if broken
    # ------------------------------------------------------------------
    emoji_col = COL["list_emojis"]
    if emoji_col not in df_filtered.columns:
        # Try to find column with "emoji" in name
        possible = [c for c in df_filtered.columns if "emoji" in c.lower()]
        if possible:
            emoji_col = possible[0]
            st.info(f"Using emoji column: `{emoji_col}`")
        else:
            st.error("No emoji column found!")
            return

    # ------------------------------------------------------------------
    # 2. Group selector
    # ------------------------------------------------------------------
    group_options = ["all whatsapp_groups"] + sorted(df_filtered[COL["group"]].unique().tolist())
    selected_group = st.selectbox("Select WhatsApp Group", group_options, key="dist_group_select")

    if selected_group == "all whatsapp_groups":
        df_emoji = df_filtered.copy()
        title_suffix = "All Groups"
    else:
        df_emoji = df_filtered[df_filtered[COL["group"]] == selected_group].copy()
        title_suffix = selected_group

    # ------------------------------------------------------------------
    # 3. PARSE list_of_all_emojis → echte Python lijst
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
        # Fallback: split by comma
        return [e.strip().strip("'\"") for e in s.split(",") if e.strip()]

    df_emoji["emojis_parsed"] = df_emoji[emoji_col].apply(safe_parse_emojis)
    df_emoji["emoji_count"] = df_emoji["emojis_parsed"].apply(len)

    df_with_emoji = df_emoji[df_emoji["emoji_count"] > 0].copy()
    if df_with_emoji.empty:
        st.warning("No valid emojis found in this group.")
        return

    # ------------------------------------------------------------------
    # 4. DEBUG: Show parsing
    # ------------------------------------------------------------------
    with st.expander("DEBUG: Parsed Emojis", expanded=False):
        st.write(f"**Messages with emojis:** {len(df_with_emoji):,}")
        sample = df_with_emoji[[emoji_col, "emojis_parsed"]].head(5)
        st.write("**Raw → Parsed:**")
        st.dataframe(sample)

    # ------------------------------------------------------------------
    # 5. Count: one per message (unique emoji)
    # ------------------------------------------------------------------
    df_with_emoji["emojis_unique"] = df_with_emoji["emojis_parsed"].apply(lambda x: list(set(x)))
    exploded = df_with_emoji.explode("emojis_unique")
    exploded = exploded[exploded["emojis_unique"].notna()]

    emoji_counts = exploded["emojis_unique"].value_counts().reset_index()
    emoji_counts.columns = ["emoji", "count"]

    # ------------------------------------------------------------------
    # 6. FINAL DEBUG
    # ------------------------------------------------------------------
    with st.expander("FINAL COUNTS (Before Filter)", expanded=True):
        st.write("**Top 20 emojis:**")
        st.dataframe(emoji_counts.head(20))
        st.write(f"**Max count:** {emoji_counts['count'].max():,}")
        st.write(f"**Total messages with emoji:** {emoji_counts['count'].sum():,}")

    # ------------------------------------------------------------------
    # 7. Filter slider
    # ------------------------------------------------------------------
    max_possible = int(emoji_counts["count"].max())
    min_count, max_count = st.slider(
        "Min–Max Messages Containing Emoji",
        min_value=0,
        max_value=max_possible,
        value=(0, max_possible),
        key="emoji_slider"
    )

    filtered = emoji_counts[
        (emoji_counts["count"] >= min_count) &
        (emoji_counts["count"] <= max_count)
    ].copy()

    if filtered.empty:
        st.warning("No emojis match filter.")
        return

    filtered = filtered.sort_values("count", ascending=False)
    show_labels = len(filtered) < 26

    # ------------------------------------------------------------------
    # 8. Plot
    # ------------------------------------------------------------------
    y_upper = filtered["count"].max() * 1.1

    fig = px.bar(
        filtered,
        x="emoji",
        y="count",
        title=f"Emoji Distribution – {title_suffix}",
        labels={"count": "Messages Containing Emoji", "emoji": "Emoji"},
        text="emoji" if show_labels else None,
        color="count",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        height=600,
        xaxis_title="Emoji",
        yaxis_title="Messages Containing Emoji",
        showlegend=False,
        yaxis=dict(range=[0, y_upper])
    )

    if show_labels:
        fig.update_traces(textposition="outside")

    st.plotly_chart(fig, use_container_width=True)