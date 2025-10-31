# src/dashboard/tabs/tab_category.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import COL, author_color


# ----------------------------------------------------------------------
# Helper: stable colour palette per WhatsApp group
# ----------------------------------------------------------------------
def _group_color_palette(df):
    groups = sorted(df[COL["group"]].unique())
    palette = px.colors.qualitative.Plotly * (len(groups) // len(px.colors.qualitative.Plotly) + 1)
    return {g: palette[i % len(palette)] for i, g in enumerate(groups)}


def render_category_tab(df_filtered):
    st.header("Category – Messages per Group / Author")

    # ------------------------------------------------------------------
    # 1. View mode
    # ------------------------------------------------------------------
    view_mode = st.radio(
        "View by",
        options=["whatsapp_group", "author"],
        format_func=lambda x: "Group" if x == "whatsapp_group" else "Author",
        horizontal=True,
        index=0,
    )

    # ------------------------------------------------------------------
    # 2. GROUP VIEW
    # ------------------------------------------------------------------
    if view_mode == "whatsapp_group":
        st.subheader("WhatsApp Group and participating Authors")

        # --------------------------------------------------------------
        # Build data: group → author → count
        # --------------------------------------------------------------
        author_group = (
            df_filtered.groupby([COL["group"], COL["author"]], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

        # Sort groups alphabetically
        group_order = sorted(author_group[COL["group"]].unique())
        author_group[COL["group"]] = pd.Categorical(
            author_group[COL["group"]], categories=group_order, ordered=True
        )

        # Sort non-AvT by count within group, AvT last
        avt_mask = author_group[COL["author"]] == "AvT"
        avt_rows = author_group[avt_mask]
        non_avt = author_group[~avt_mask]

        non_avt = non_avt.sort_values(
            [COL["group"], "count"], ascending=[True, False]
        ).reset_index(drop=True)

        # Rebuild: non-AvT first, then AvT
        final_rows = []
        for group in group_order:
            grp_non = non_avt[non_avt[COL["group"]] == group]
            grp_avt = avt_rows[avt_rows[COL["group"]] == group]
            final_rows.append(grp_non)
            if not grp_avt.empty:
                final_rows.append(grp_avt)
        author_group = pd.concat(final_rows, ignore_index=True)

        # Create x-label: Group<br>Author
        author_group["x_label"] = (
            author_group[COL["group"]].astype(str) + "<br>" + author_group[COL["author"]]
        )

        # Insert single invisible spacer between groups
        spaced_data = []
        current_group = None
        spacer_idx = 0
        for _, row in author_group.iterrows():
            if current_group is not None and row[COL["group"]] != current_group:
                spacer_idx += 1
                spaced_data.append({
                    "x_label": f"__spacer__{spacer_idx}",
                    "count": 0,
                    "bar_color": "rgba(0,0,0,0)",
                    "show_text": False,
                    COL["group"]: "SPACER",
                    COL["author"]: ""
                })
            row_dict = row.to_dict()
            row_dict["show_text"] = True
            spaced_data.append(row_dict)
            current_group = row[COL["group"]]

        author_group = pd.DataFrame(spaced_data)

        # Color: AvT = white, others = group color
        group_palette = _group_color_palette(df_filtered)
        def get_color(row):
            if row[COL["author"]] == "AvT":
                return "#ffffff"
            elif row[COL["group"]] == "SPACER":
                return "rgba(0,0,0,0)"
            else:
                return group_palette.get(row[COL["group"]], "#cccccc")
        author_group["bar_color"] = author_group.apply(get_color, axis=1)

        # Plot
        fig = px.bar(
            author_group,
            x="x_label",
            y="count",
            color="bar_color",
            color_discrete_map="identity",
            text=author_group["count"].where(author_group["show_text"], ""),
            labels={"count": "Messages"},
        )

        fig.update_traces(
            texttemplate="%{text}",
            textposition="outside",
            marker_line=dict(width=2, color="black"),
        )

        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=author_group["x_label"].tolist(),
            tickmode="array",
            tickvals=[x for x in author_group["x_label"] if not x.startswith("__spacer__")],
            ticktext=[x.split("<br>")[-1] for x in author_group["x_label"] if not x.startswith("__spacer__")],
            title="WhatsApp Group → Author",
        )

        fig.update_layout(
            barmode="stack",
            height=700,
            yaxis_title="Messages",
            showlegend=False,
            margin=dict(l=60, r=60, t=80, b=120),
            bargap=0.1,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Optional: Add a small note
        st.caption("Authors ordered by message volume within each group. AvT shown last. Groups separated for clarity.")

    # ------------------------------------------------------------------
    # 3. AUTHOR VIEW – now ONLY shows total messages per author
    # ------------------------------------------------------------------
    else:
        st.subheader("Total Messages per Author (All Groups Combined)")

        total = df_filtered[COL["author"]].value_counts().reset_index()
        total.columns = [COL["author"], "count"]  # ← FIXED LINE

        # Sort: non-AvT descending, AvT last
        avt_mask = total[COL["author"]] == "AvT"
        avt_row = total[avt_mask]
        non_avt = total[~avt_mask].sort_values("count", ascending=False)
        total_sorted = pd.concat([non_avt, avt_row], ignore_index=True) if not avt_row.empty else non_avt

        total_sorted["color"] = total_sorted[COL["author"]].apply(author_color)

        fig_total = px.bar(
            total_sorted,
            x=COL["author"],
            y="count",
            color="color",
            color_discrete_map="identity",
            text="count",
            title="All Groups Combined",
            labels={"count": "Messages"},
        )
        fig_total.update_traces(
            textposition="outside",
            marker_line=dict(width=1.5, color="black"),
        )
        fig_total.update_layout(
            showlegend=False,
            height=600,
            xaxis_title="Author",
            yaxis_title="Total Messages",
            bargap=0.2,
        )
        if total_sorted[COL["author"]].nunique() > 15:
            fig_total.update_xaxes(tickangle=45)

        st.plotly_chart(fig_total, use_container_width=True)

        st.caption("Authors sorted by total message count across all groups. AvT shown last.")

    # ------------------------------------------------------------------
    # 4. Summary stats (unchanged)
    # ------------------------------------------------------------------
    with st.expander("Summary Stats"):
        total_msgs = len(df_filtered)
        unique_authors = df_filtered[COL["author"]].nunique()
        unique_groups = df_filtered[COL["group"]].nunique()
        st.write(f"**Total Messages:** {total_msgs:,}")
        st.write(f"**Unique Authors:** {unique_authors}")
        st.write(f"**Unique Groups:** {unique_groups}")