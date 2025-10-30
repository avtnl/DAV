# tabs/tab_multi_dimensions.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import chi2
from utils.style_analyzer import compute_style_fingerprint, compute_message_fingerprint
from config import COL, GROUP_COLORS


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert hex (#1f77b4) → rgba(31,119,180,0.2)"""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_ellipse_points(mean_x, mean_y, width, height, angle, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    x = mean_x + (width / 2) * np.cos(theta) * cos_a - (height / 2) * np.sin(theta) * sin_a
    y = mean_y + (width / 2) * np.cos(theta) * sin_a + (height / 2) * np.sin(theta) * cos_a
    return x, y


def render_multi_dimensions_tab(df: pd.DataFrame):
    st.header("Multi Dimensions – Fingerprint")

    fingerprint_type = st.radio("Fingerprint Type", ["Style", "Message"], horizontal=True, index=0)

    if fingerprint_type == "Style":
        agg = compute_style_fingerprint(df.copy())
    else:
        agg = compute_message_fingerprint(df.copy())

    if agg.empty:
        st.warning("No data after filtering.")
        return

    min_y, max_y = int(agg[COL["year"]].min()), int(agg[COL["year"]].max())
    year_range = st.slider("Year range", min_y, max_y, (min_y, max_y))
    agg = agg[agg[COL["year"]].between(year_range[0], year_range[1])]

    view_mode = st.radio("View", ["WhatsApp Group", "Author"], horizontal=True, index=0)
    show_ellipses = st.checkbox("Show ellipses", value=False)
    conf_level = st.slider("Confidence level (%)", 0, 80, 50, step=5) if show_ellipses else 50

    # --- Color mapping ---
    if view_mode == "WhatsApp Group":
        def assign_group(row):
            if row[COL["author"]] == "AvT":
                return "AvT"
            return row["whatsapp_group_temp"]
        agg["plot_group"] = agg.apply(assign_group, axis=1)
        color_col = "plot_group"
        color_map = {
            "maap": "#1f77b4", "dac": "#ff7f0e", "golfmaten": "#2ca02c",
            "tillies": "#7f7f7f", "AvT": "#d62728"
        }
    else:
        color_col = COL["author"]
        color_map = None

    fig = px.scatter(
        agg,
        x="tsne_x", y="tsne_y",
        size="msg_count",
        color=color_col,
        color_discrete_map=color_map,
        hover_data={COL["author"]: True, COL["year"]: True, "msg_count": True},
        labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2"},
        title=f"{fingerprint_type} Fingerprint – {view_mode} View"
    )

    # --- Add ellipses ---
    if show_ellipses:
        chi_val = np.sqrt(chi2.ppf(conf_level / 100, df=2))
        for group in agg[color_col].unique():
            sub = agg[agg[color_col] == group]
            if len(sub) < 2:
                continue
            x, y = sub['tsne_x'].values, sub['tsne_y'].values
            cov = np.cov(x, y)
            mean_x, mean_y = np.mean(x), np.mean(y)
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            width, height = 2 * lambda_[0] * chi_val, 2 * lambda_[1] * chi_val
            angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
            ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)
            color = color_map.get(group, px.colors.qualitative.Plotly[0]) if color_map else "#1f77b4"
            fig.add_trace(go.Scatter(
                x=ell_x, y=ell_y,
                mode='lines',
                fill='toself',
                fillcolor=hex_to_rgba(color, 0.2),   # ← CORRECT
                line=dict(color=color, width=2),
                name=f"{group} {conf_level}%",
                showlegend=False,
                hoverinfo='skip'
            ))

    st.plotly_chart(fig, use_container_width=True)