# tabs/tab_relationships.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from config import COL, GROUP_COLORS


def render_relationships_tab(df: pd.DataFrame) -> None:
    """Relationships tab – Graph (bubble) or Matrix (correlation heatmap)."""
    st.header("Relationships")

    # ------------------------------------------------------------------
    # 1. Radio: Graph vs Matrix
    # ------------------------------------------------------------------
    view_mode = st.radio(
        "View mode",
        options=["Graph", "Matrix"],
        horizontal=True,
        index=0,  # default = Graph
        key="rel_view_mode",
    )

    # ------------------------------------------------------------------
    # 2. Metric → column mapping (used in both views)
    # ------------------------------------------------------------------
    metrics = {
        "Message Length": COL["length_chat"],
        "Number of Words": COL["num_words"],
        "Number of Punctuations": COL["num_punct"],
        "Number of Capitals": COL["num_capitals"],
        "Number of Emojis": COL["num_emojis"],
        "Number of Numbers": COL["num_numbers"],
    }

    # ------------------------------------------------------------------
    # 3. Aggregation: per group + author
    # ------------------------------------------------------------------
    agg_dict = {
        "num_messages": pd.NamedAgg(column=COL["timestamp"], aggfunc="count"),
    }
    for _label, col in metrics.items():
        agg_dict[f"avg_{col}"] = pd.NamedAgg(column=col, aggfunc="mean")

    grouped = df.groupby([COL["group"], COL["author"]], as_index=False).agg(**agg_dict)

    # ------------------------------------------------------------------
    # 4. VIEW: GRAPH (Bubble Chart)
    # ------------------------------------------------------------------
    # -------------------------- GRAPH --------------------------
    if view_mode == "Graph":
        st.subheader("Bubble Chart: Author Style Relationships")

        col_x, col_y = st.columns(2)
        with col_x:
            x_label = st.selectbox("X-axis", list(metrics.keys()), index=1, key="rel_x_axis")
        with col_y:
            y_label = st.selectbox("Y-axis", list(metrics.keys()), index=2, key="rel_y_axis")

        if x_label == y_label:
            st.warning("X-axis and Y-axis cannot be the same!")
            st.info("**Start again** – choose **different** metrics.")
            if st.button("Start again", type="primary"):
                st.session_state.pop("rel_x_axis", None)
                st.session_state.pop("rel_y_axis", None)
                st.experimental_rerun()
            st.stop()

        # --- Compute correlation matrix once (same as Matrix tab) ---
        pivot = grouped.pivot_table(
            index=COL["author"], values=[f"avg_{col}" for col in metrics.values()], aggfunc="first"
        ).fillna(0)
        corr = pivot.corr(method="pearson")
        rename_dict = {f"avg_{col}": label for label, col in metrics.items()}
        corr_renamed = corr.rename(columns=rename_dict, index=rename_dict)

        # --- Extract the specific correlation ---
        pearson_r = corr_renamed.loc[y_label, x_label]  # row=Y, col=X
        pearson_r = round(pearson_r, 3)

        # --- Display correlation above the chart ---
        st.markdown(
            f"""
            <div style="text-align: center; padding: 10px; background-color: #000000; border-radius: 8px; margin: 15px 0;">
                <strong>Applicable Pearson Correlation:</strong>
                <span style="font-size: 1.4em; color: {"#d62728" if pearson_r < 0 else "#2ca02c" if pearson_r > 0.7 else "#1f77b4"};">
                    {pearson_r:+.3f}
                </span>
                <span style="font-size: 0.9em; color: #666;">
                    &nbsp; (between <i>{x_label}</i> and <i>{y_label}</i>)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Bubble chart ---
        x_col = f"avg_{metrics[x_label]}"
        y_col = f"avg_{metrics[y_label]}"

        fig = px.scatter(
            grouped,
            x=x_col,
            y=y_col,
            size="num_messages",
            color=COL["group"],
            color_discrete_map=GROUP_COLORS,
            hover_data={COL["author"]: True, "num_messages": True},
            labels={
                x_col: f"{x_label} (avg)",
                y_col: f"{y_label} (avg)",
                "num_messages": "Messages",
            },
        )

        ols = px.scatter(grouped, x=x_col, y=y_col, trendline="ols")
        trend_line = go.Scatter(
            x=ols.data[1].x,
            y=ols.data[1].y,
            mode="lines",
            line={"color": "red", "dash": "dash", "width": 2},
            name="Trend (OLS)",
            showlegend=True,
        )
        fig.add_trace(trend_line)

        fig.update_layout(
            xaxis_title=f"{x_label} (avg per message)",
            yaxis_title=f"{y_label} (avg per message)",
            legend_title="WhatsApp Group",
            height=650,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 5. VIEW: MATRIX (Correlation Heatmap)
    # ------------------------------------------------------------------
    else:  # view_mode == "Matrix"
        st.subheader("Correlation Matrix: Message Style Features")

        # Pivot: one row per author, columns = avg metrics
        pivot = grouped.pivot_table(
            index=COL["author"],
            values=[f"avg_{col}" for col in metrics.values()],
            aggfunc="first",  # each author appears once
        ).fillna(0)

        # Compute correlation matrix
        corr = pivot.corr(method="pearson")

        # Rename columns to human-readable
        rename_dict = {f"avg_{col}": label for label, col in metrics.items()}
        corr_renamed = corr.rename(columns=rename_dict, index=rename_dict)

        # Plotly heatmap
        fig = go.Figure(
            data=go.Heatmap(
                z=corr_renamed.values,
                x=corr_renamed.columns,
                y=corr_renamed.index,
                colorscale="RdBu",
                zmid=0,
                text=corr_renamed.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 16},
                hoverongaps=False,
            )
        )

        fig.update_layout(
            title="Pearson Correlation of Message Style Features (per Author)",
            xaxis_title="Feature (average per message)",
            yaxis_title="Feature (average per message)",
            height=600,
            width=700,
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("How to read this matrix"):
            st.markdown("""
            - **1.0** = perfect positive correlation (e.g., more words → more punctuation)
            - **-1.0** = perfect negative correlation
            - **0** = no linear relationship
            - Values are **averages per author** across all their messages
            """)
