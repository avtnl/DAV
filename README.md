DAV - WhatsApp Chat Analyzer

## License

License: MIT Python 3.12+
PyPI version CI

---

## Author

- Name: Anthony van Tilburg 
- Student Number: 1905780
- git: https://github.com/avtnl/DAV
- Institution: Hogeschool Utrecht (Netherlands) 
- Course: Master of Informatics | Applied Data Science | Data Analysis & Visualisation (DAV)

---

## Description

DAV - WhatsApp Chat Analyzer is a Python package for deep analysis of exported WhatsApp chat logs.
It transforms raw .txt exports into rich, cleaned datasets with 60+ engineered features (emojis, response times, punctuation, attachments, etc.), and generates 5 publication-ready visualizations plus an interactive Streamlit dashboard.
Built for group behavior insights, from seasonal activity in golf chats to emoji distributions and multi-dimensional message styles.


---

## Features

Feature				          Description
=====================   ====================================================================
Preprocessing			      Clean raw TXT exports, handle media, deletions, system messages
60+ Features			      Emoji lists, response times, word length, punctuation, attachments
6 Visualizations		    Categories, Time Series, Distribution, Relationships, Multi Dimensions"
Seasonality Proof		    ACF, Fourier, Decomposition, Filtered signals (Script 2)"
Power-Law Fits		      Emoji frequency follows Zipf (Script 3)
Interactive Dashboard	  Streamlit app with all plots and data
Caching & Reuse		      Skip preprocessing with reuse_whatsapp_all = true
Modular Pipeline		    Run any subset of scripts (0–6)
Logging				          Timestamped logs in logs/T

## Folders and Files

DAV-whatsapp-analyzer/
├── data/                       # Data
│   ├── raw                     # Clean raw TXT exports of whatsapp data
│   └── processed               # Pre-processed whatsapp data in csv and parq format & generated and reusable enriched (core) datafile
├── img                         # Contains generated images
├── src/                        # Code
│   ├── dashboard/              # Code related to Dashboard
│   │   ├── tabs/               # Code, contains more modules
│   │   ├── utils/              # Code, contains more modules
│   │   ├── config.py           # Code
│   │   ├── data_loader         # Code
│   │   ├── streamlit_app       # Code
│   │   ├── your_whatsapp_data  # Copy of generated and reusable enriched (core) datafile
│   ├── data_editor/            # Code related to Data_Editor
│   │   ├── __init__.py         # Code
│   │   ├── cleaners.py         # Code
│   │   ├── core.py             # Code
│   │   ├── features.py         # Code
│   │   ├── utilities.py        # Code
│   ├── scripts/                # Code, contains more modules | Individual scripts triggered by input of main.py
│   ├── style_output/           # Data generate by Script 6 (Multi_Dimensions) and partly source material for Dashboard 
│   ├── __init__.py             
│   ├── constants.py            # Constants used in code
│   ├── data_preparation.py     # Code
│   ├── file_manager.py         # Code
│   ├── main.py                 # Code and main script - here you are able to modify the Scripts (input)
│   ├── plotmanager.py          # Code
├── tables/                     # Contains generated tables
├── .gitignore                  # GIT related
├── config.toml                 # Configuration file, mainly to control input-files, pre-processing and re-use of the enriched data file
├── hello.py                    # Simple module for testing imports (created by {UV})
├── pyproject.toml              # Package dependencies
└── README.md                   # THis file - Project overview and author info

---

## Outputs

Type			Location
========= ================================
Logs			logs/logfile-YYYYMMDD-HHMMSS.log
Data			data/processed/ (Parquet + CSV)
Plots		  img/ (timestamped PNGs)
Tables		tables/ (CSV)
Dashboard	http://localhost:8501

---

## Usage

1. Configure (config.toml)

2. Run Pipeline (main.py) by using: python src/main.py
   Example: SCRIPTS = [1, 2, 3, 4, 5, 6]  # To run all script, you can ender any sub selection.

   Script	Output					        Purpose
   ======  =====================   ======================================== 
   0		    data/processed/*.parq		Preprocess raw TXT to enriched DataFrame
   1		    img/categories-*.png		Messages by group/author
   2		    img/time-*.png			    Weekly heartbeat + seasonality evidence
   3		    img/distribution-*.png	Emoji frequency + power-law
   4		    img/bubble-*.png			  Words vs. punctuation
   5		    img/multidim-*.png		  PCA/ T-SNE + HDBSCAN clustering
   6		    http://localhost:8501		Streamlit dashboard
            Press ENTER in terminal to exit dashboard.

   Script 5
   ========
   - For script 5 (Multi-Dimensions) there is an additional constant SCRIPT_5_DETAILS.
     Example: SCRIPT_6_DETAILS = ["tsne", True, 0, 75, True, True, 3]
   - Parameters:  
     1 "tsne"  plot_type                    "both", "pca", "tsne"
     2 True    by_group                     True = Focus on Whatsapp Group, False = Focus on Individual Authors
     3 0       draw ellipses                0 = no elippses, 1 = max 1 ellips per group/author, 2 = max 3 mini ellipses per group/author
     4 75      confidence level (ellipses)  Value between 0 and 100 (percentages)
     5 True    Use embedded model           True = Uses model, see WARNING
     6 True    Hybrid                       Works for Embedded Model ID = 1 (combines 25 style features with model)
     7 3       Embedded Model ID            1 = Anna-Wiegman (style based), 2 = Minilm (scentence based), 3 = all-mpnet-base (sentence)
  
   - WARNING: If, for script 5 (multi dimensions), the parameter for using a ‘transformer’ is set to ‘True’:it will take several minutes
     before the desired image finally appears! Some patience is required! Progress is shown in the
     ‘terminal output’ so you know roughly how much longer it will take.

   The images belonging to scripts 1 through 4 are displayed automatically. By clicking away a displayed image (the ‘X’ in the top-right corner), you trigger the continuation of the script (if applicable). You can click away the image again to continue the script further, and so on.

   The image belonging to script 5 (multi dimensions) is NOT displayed automatically. To actually view it, you need to go to the ‘img’ folder and look for the correct file name (stating with 'Style_').

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




