DAV - WhatsApp Chat Analyzer Dashboard

## License

License: MIT Python 3.12+
PyPI version CI

---

## Author

- Name: Anthony van Tilburg 
- Student Number: 1905780
- Institution: Hogeschool Utrecht (Netherlands) 
- Course: Master of Informatics | Applied Data Science | Data Analysis & Visualisation (DAV)

---

## Description

An interactive dashboard built with Streamlit for analyzing exported WhatsApp chat data. It provides insights into message categories, time-based patterns, emoji distributions, author relationships, and multi-dimensional style/message fingerprints.

This dashboard processes a pre-enriched CSV file (e.g., from a WhatsApp export pipeline) and visualizes key metrics using Plotly. It's designed for personal data exploration, such as understanding communication patterns in group chats.


---

## Features

- Category Tab:         Visualize messages per WhatsApp group and author, with bar charts showing participation
                        Supports grouping by group or author, with custom sorting (e.g., "AvT" always last).
- Time Tab:             Average messages per week of the year, filtered by group. Helps identify seasonal or weekly trends.
- Distribution Tab:     Emoji usage overview with likelihood and cumulative probabilities. Includes a "Test Probability" mode for
                        simulating emoji occurrence in samples.
- Relationships Tab:    Explore correlations between message styles (e.g., length, words, emojis) via bubble charts or
                        Pearson correlation heatmaps.
- Multi Dimensions Tab: Advanced fingerprint analysis for style (punctuation, rhythm) or message content (semantics). Uses t-SNE
                        for visualization, with optional ellipses for confidence intervals and GMM clustering.

Filters (year range, message type, length) are applied via the sidebar and affect all tabs.

---

## Folders and Files

DAV-whatsapp-analyzer/
├── src/
│   └── dashboard/
│       ├── streamlit_app.py            # Main Streamlit app entry point
│       ├── data_loader.py              # Loads and preprocesses CSV data
│       ├── config.py                   # Column names and color configs
│       ├── utils/
│       │   ├── filters.py              # Sidebar filter logic
│       │   └── style_analyzer.py       # Fingerprint computation (t-SNE, PCA, GMM)
│       └── tabs/
│           ├── tab_category.py         # Category tab rendering
│           ├── tab_time.py             # Time tab rendering
│           ├── tab_distribution.py     # Distribution tab rendering
│           ├── tab_relationships.py    # Relationships tab rendering
│           └── tab_multi_dimensions.py # Multi Dimensions tab rendering
├── your_whatsapp_data.csv              # Example enriched CSV (place your file here)
├── pyproject.toml                      # Project config for dependencies
├── requirements.txt                    # Fallback dependencies (if not using pyproject.toml)
├── .gitignore
└── README.md                           # This file - Project overview and author info

---

## Tech Stack (2025 Tilburg-proof)

- Python                        ≥ 3.12  
- Streamlit                     The complete dashboard uses Streamlit (v1.38+ tested)  
- Plotly                        All interactive graphs  
- Pandas, scikit-learn, scipy   For thefingerprints en statistics

---

## Installation & Setup

1. Prerequisites
- Python >= 3.12
- Git

2. Clone and enter the project
git clone https://github.com/avtnl/DAV
cd dav

3. Create and activate a virtual environment (recommended)
OPTION 3a – Using venv (classic & works everywhere)
bash: python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# or
.venv\Scripts\activate         # Windows PowerShell

OPTION 3b – Using uv (super fast – very popular)
bash: uv venv .venv
source .venv/bin/activate      # or just use `uv run` later, no need to activate!

4. Install the project
Normal installation (for regular use)
bash: pip install -e .
# or even faster with uv:
uv pip install -e .




