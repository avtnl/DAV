# tabs/tab_multi_dimensions.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from utils.style_analyzer import compute_style_fingerprint, compute_message_fingerprint
from config import COL
import os


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
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

    # === 1. Fingerprint Type ===
    fingerprint_type = st.radio(
        "Fingerprint Type",
        ["Style", "Message"],
        horizontal=True,
        help="**Style** = how people write. **Message** = what they say."
    )

    # === 2. Model Options (Simplified: no hybrid light-style) ===
    if fingerprint_type == "Style":
        model_options = {
            "My Model": "style_output/style_summary_my_model.csv",
            "AnnaWegmann (classic)": "style_output/style_summary_anna_classic.csv",
            "AnnaWegmann + MyModel (fused)": "style_output/style_summary_anna_plus_my_model.csv",
        }
        help_text = "Style models capture author identity via punctuation, emojis, rhythm."
    else:
        model_options = {
            "Minilm": "message_output/message_summary_minilm.csv",
            "all-mpnet-base": "message_output/message_summary_mpnet.csv",
        }
        help_text = "Message models cluster by semantic content (topics, meaning)."

    selected_model = st.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        help=help_text,
        key=f"model_select_{fingerprint_type.lower()}"
    )
    pregen_path = model_options[selected_model]

    # === 3. Load Data ===
    if os.path.exists(pregen_path):
        agg = pd.read_csv(pregen_path)
        st.success(f"Loaded: **{selected_model}**")
    else:
        func = compute_style_fingerprint if fingerprint_type == "Style" else compute_message_fingerprint
        with st.spinner(f"Computing {fingerprint_type.lower()} fingerprint..."):
            agg = func(df.copy())
        st.info("Computed on-the-fly")

    if agg.empty:
        st.warning("No data after filtering.")
        return

    # === 4. Advanced Visualization (Flat Layout) ===
    st.markdown("### Advanced Visualization")

    col1, col2 = st.columns([1, 2])

    with col1:
        # View: Group vs Author
        view_mode = st.radio(
            "View",
            ["Group", "Author"],
            horizontal=True,
            key=f"view_mode_{fingerprint_type.lower()}"
        )

        # Show Ellipses
        show_ellipses = st.checkbox("Show Ellipses", value=False)

    with col2:
        # Confidence slider (only active when ellipses on)
        conf_level = st.slider(
            "Confidence level (%)",
            0, 80, 50, step=5,
            disabled=not show_ellipses,
            help="Larger % = larger ellipse"
        )

    st.markdown("---")

    # === 5. Color Mapping ===
    if view_mode == "Group":
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

    # === 6. Hover Data (Safe) ===
    base_hover = [COL["author"], COL["year"], "msg_count"]
    if "plot_group" in agg.columns:
        base_hover.append("plot_group")

    # === 7. Scatter Plot ===
    fig = px.scatter(
        agg,
        x="tsne_x", y="tsne_y",
        size="msg_count",
        color=color_col,
        color_discrete_map=color_map,
        hover_data=base_hover,
        labels={"tsne_x": "t-SNE Dimension 1", "tsne_y": "t-SNE Dimension 2"},
        title=f"{fingerprint_type} Fingerprint – {selected_model}"
    )

    # # === 8. Add Ellipses ===
    # if show_ellipses:
    #     chi_val = np.sqrt(chi2.ppf(conf_level / 100, df=2))
    #     for group in agg[color_col].unique():
    #         sub = agg[agg[color_col] == group]
    #         if len(sub) < 2:
    #             continue
    #         x, y = sub['tsne_x'].values, sub['tsne_y'].values
    #         cov = np.cov(x, y)
    #         mean_x, mean_y = np.mean(x), np.mean(y)
    #         lambda_, v = np.linalg.eig(cov)
    #         lambda_ = np.sqrt(lambda_)
    #         width, height = 2 * lambda_[0] * chi_val, 2 * lambda_[1] * chi_val
    #         angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    #         ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)

    #         color = color_map.get(group, px.colors.qualitative.Plotly[0]) if color_map else "#1f77b4"
    #         fig.add_trace(go.Scatter(
    #             x=ell_x, y=ell_y,
    #             mode='lines',
    #             fill='toself',
    #             fillcolor=hex_to_rgba(color, 0.2),
    #             line=dict(color=color, width=2),
    #             name=f"{group} {conf_level}%",
    #             showlegend=False,
    #             hoverinfo='skip'
    #         ))


    # === MODIFIED ELLIPSE DRAWING WITH CLUSTERING ===
    if show_ellipses:
        chi_val = np.sqrt(chi2.ppf(conf_level / 100, df=2))
        
        for group in agg[color_col].unique():
            sub = agg[agg[color_col] == group].copy()
            if len(sub) < 3:  # Need at least 3 points for meaningful GMM
                continue

            x = sub['tsne_x'].values.reshape(-1, 1)
            y = sub['tsne_y'].values.reshape(-1, 1)
            X = np.hstack([x, y])

            # === STEP 1: Fit GMM to detect sub-clusters ===
            # Try up to 3 components; use BIC to avoid overfitting
            best_gmm = None
            best_bic = np.inf
            best_n = 1

            for n in range(1, min(4, len(sub))):  # max 3 clusters
                gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_n = n

            # === STEP 2: Draw one ellipse per GMM component ===
            for i in range(best_n):
                weight = best_gmm.weights_[i]
                mean = best_gmm.means_[i]
                cov = best_gmm.covariances_[i]

                # Skip very small components
                if weight < 0.1:
                    continue

                mean_x, mean_y = mean[0], mean[1]

                # Eigen decomposition
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                width = 2 * lambda_[0] * chi_val
                height = 2 * lambda_[1] * chi_val
                angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))

                ell_x, ell_y = get_ellipse_points(mean_x, mean_y, width, height, angle)

                color = color_map.get(group, px.colors.qualitative.Plotly[0]) if color_map else "#1f77b4"
                opacity = 0.2 + 0.3 * weight  # Optional: larger clusters more opaque

                fig.add_trace(go.Scatter(
                    x=ell_x, y=ell_y,
                    mode='lines',
                    fill='toself',
                    fillcolor=hex_to_rgba(color, opacity),
                    line=dict(color=color, width=2),
                    name=f"{group} Cluster {i+1} {conf_level}%",
                    showlegend=False,
                    hoverinfo='skip'
                ))


    # === 9. ZOOM IN + ALL GRIDLINES BOLD & BLACK ===
    y_min, y_max = agg['tsne_y'].min(), agg['tsne_y'].max()
    y_center = (y_min + y_max) / 2
    y_half_span = (y_max - y_min) / 2
    new_half_span = y_half_span * 1

    fig.update_layout(
        yaxis=dict(
            range=[y_center - new_half_span, y_center + new_half_span],
            title="t-SNE Dimension 2",
            showgrid=False,
            gridcolor="lightgray",
            gridwidth=0,
            zeroline=False,
            zerolinecolor="lightgray",
            zerolinewidth=1
        ),
        xaxis=dict(
            title="t-SNE Dimension 1",
            showgrid=False,
            gridcolor="lightgray",
            gridwidth=0,
            zeroline=False,
            zerolinecolor="lightgray",
            zerolinewidth=1
        ),
        height=600,
        margin=dict(l=60, r=20, t=40, b=40)
    )

    # === 10. DISPLAY PLOT ===
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})