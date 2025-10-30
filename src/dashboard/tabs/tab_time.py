# src/dashboard/tabs/tab_time.py
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from config import COL

def render_time_tab(df_filtered):
    st.header("Time – Avg Messages per Week")

    # ------------------------------------------------------------------
    # 1. Group selector (only one group)
    # ------------------------------------------------------------------
    groups = sorted(df_filtered[COL["group"]].unique().tolist())
    selected_group = st.selectbox("Select WhatsApp Group", groups, key="time_group_select")

    # ------------------------------------------------------------------
    # 2. Filter by selected group
    # ------------------------------------------------------------------
    df_group = df_filtered[df_filtered[COL["group"]] == selected_group].copy()

    # ------------------------------------------------------------------
    # 3. Use year range from SIDEBAR (no local slider!)
    # ------------------------------------------------------------------
    # df_filtered is already filtered by year from sidebar
    # So we just use it directly
    df_final = df_group.copy()

    # ------------------------------------------------------------------
    # 4. Compute average per week (1–53)
    # ------------------------------------------------------------------
    if len(df_final) == 0:
        st.warning("No messages in selected filters.")
        return

    weekly_counts = df_final.groupby(COL["week"]).size()
    years_in_data = df_final[COL["year"]].nunique()
    avg_per_week = weekly_counts / years_in_data

    # Fill missing weeks
    avg_per_week = avg_per_week.reindex(range(1, 54), fill_value=0.0)

    # ------------------------------------------------------------------
    # 5. Plot
    # ------------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, 54)),
        y=avg_per_week.values,
        mode='lines+markers',
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6),
        name="Avg Messages"
    ))

    # Get year range from data
    year_min = df_final[COL["year"]].min()
    year_max = df_final[COL["year"]].max()

    fig.update_layout(
        title=f"Average Messages per Week – {selected_group}<br>"
              f"Years: {year_min}–{year_max} ({years_in_data} year(s))",
        xaxis_title="Week of Year",
        yaxis_title="Avg Messages",
        xaxis=dict(tickmode='linear', tick0=1, dtick=1),
        height=600,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    with st.expander("Summary"):
        total_msgs = len(df_final)
        st.write(f"**Group:** {selected_group}")
        st.write(f"**Years:** {year_min} – {year_max}")
        st.write(f"**Total Messages:** {total_msgs:,}")
        st.write(f"**Average per Week (53 weeks):** {avg_per_week.mean():.2f}")