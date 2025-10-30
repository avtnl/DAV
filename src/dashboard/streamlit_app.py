# src/dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import os
from data_loader import load_csv
from utils.filters import sidebar_filters
from tabs.tab_time import render_time_tab
from tabs.tab_distribution import render_distribution_tab
from tabs.tab_relationships import render_relationships_tab
from tabs.tab_multi_dimensions import render_multi_dimensions_tab

# ----------------------------------------------------------------------
# Global CSS for all Plotly charts
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .js-plotly-plot .plotly, .js-plotly-plot {
        height: 800px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="WhatsApp Dashboard", layout="wide")

# ----------------------------------------------------------------------
# 1. CSV INPUT – either upload or use a local file
# ----------------------------------------------------------------------
DEFAULT_CSV = "your_whatsapp_data.csv"          # <-- change if you keep it locally
LOCAL_PATH = os.path.join(os.path.dirname(__file__), DEFAULT_CSV)

# Try local file first
if os.path.exists(LOCAL_PATH):
    csv_path = LOCAL_PATH
    st.success(f"Found local CSV: `{DEFAULT_CSV}`")
else:
    st.warning("Local CSV not found – please upload your file.")
    uploaded = st.file_uploader(
        "Upload your WhatsApp export CSV",
        type=["csv"],
        help="The file must contain the columns described in the spec."
    )
    if uploaded is None:
        st.stop()                     # stop execution until a file is uploaded
    csv_path = uploaded

# ----------------------------------------------------------------------
# 2. Load data (cached)
# ----------------------------------------------------------------------
# `load_csv` works with a file-path **or** a file-like object
df_raw = load_csv(csv_path)

# ----------------------------------------------------------------------
# 3. Sidebar filters (shared by every tab)
# ----------------------------------------------------------------------
filters = sidebar_filters(df_raw)
df = filters["df"]          # filtered DataFrame ready for plots

# ----------------------------------------------------------------------
# 4. Tabs
# ----------------------------------------------------------------------
from tabs.tab_category import render_category_tab   # <-- NEW IMPORT

tab_category, tab_time, tab_dist, tab_rel, tab_multi = st.tabs(
    ["Category", "Time", "Distribution", "Relationships", "Multi Dimensions"]
)

# -------------------------- CATEGORY --------------------------
with tab_category:
    render_category_tab(df)          # <-- FULLY IMPLEMENTED

# -------------------------- TIME --------------------------
with tab_time:
    render_time_tab(df)

# -------------------------- DISTRIBUTION --------------------------
with tab_dist:
    render_distribution_tab(df)

# -------------------------- RELATIONSHIPS --------------------------
with tab_rel:
    render_relationships_tab(df)

# -------------------------- MULTI DIMENSIONS --------------------------
with tab_multi:
    render_multi_dimensions_tab(df)