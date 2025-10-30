# src/dashboard/tabs/tab_category.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import COL, author_color

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
        index=0
    )

    # ------------------------------------------------------------------
    # 2. GROUP VIEW → Stacked bar (AvT blue, others light gray)
    # ------------------------------------------------------------------
    if view_mode == "whatsapp_group":
        st.subheader("Messages per Author (by Group)")

        author_group = (
            df_filtered.groupby([COL["group"], COL["author"]])
            .size()
            .reset_index(name="count")
        )
        author_group = author_group.sort_values([COL["group"], COL["author"]])
        author_group["color"] = author_group[COL["author"]].apply(author_color)

        fig = go.Figure()
        for author in author_group[COL["author"]].unique():
            data = author_group[author_group[COL["author"]] == author]
            fig.add_trace(go.Bar(
                x=data[COL["group"]],
                y=data["count"],
                name=author,
                marker_color=author_color(author)
            ))

        fig.update_layout(
            barmode='stack',
            height=600,
            xaxis_title="WhatsApp Group",
            yaxis_title="Total Messages",
            legend_title="Author",
            title="Messages per Author (by Group)"
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Detailed Table"):
            pivot = author_group.pivot(index=COL["author"], columns=COL["group"], values="count").fillna(0)
            st.dataframe(pivot)

    # ------------------------------------------------------------------
    # 3. AUTHOR VIEW → Select group + light gray bars
    # ------------------------------------------------------------------
    else:
        st.subheader("Messages in Selected Group")

        groups = ["all whatsapp_groups"] + sorted(df_filtered[COL["group"]].unique().tolist())
        selected_group = st.selectbox("Select Group", groups, key="author_group_select")

        if selected_group == "all whatsapp_groups":
            subset = df_filtered
            title = "All Groups Combined"
        else:
            subset = df_filtered[df_filtered[COL["group"]] == selected_group]
            title = selected_group

        author_counts = subset[COL["author"]].value_counts().reset_index()
        author_counts.columns = [COL["author"], "count"]
        author_counts["color"] = author_counts[COL["author"]].apply(author_color)

        fig = px.bar(
            author_counts,
            x=COL["author"],
            y="count",
            color="color",
            color_discrete_map={
                "#1f77b4": "#1f77b4",   # AvT → blue
                "#f0f0f0": "#f0f0f0"    # others → light gray
            },
            title=title,
            labels={"count": "Messages"}
        )
        fig.update_layout(showlegend=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # 4. Summary stats
    # ------------------------------------------------------------------
    with st.expander("Summary Stats"):
        total_msgs = len(df_filtered)
        unique_authors = df_filtered[COL["author"]].nunique()
        unique_groups = df_filtered[COL["group"]].nunique()
        st.write(f"**Total Messages:** {total_msgs:,}")
        st.write(f"**Unique Authors:** {unique_authors}")
        st.write(f"**Unique Groups:** {unique_groups}")