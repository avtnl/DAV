## Author

- Name: Anthony van Tilburg 
- Student Number: 1905780
- Institution: Hogeschool Utrecht (Netherlands) 
- Course: Master of Informatics | Applied Data Science | Data Analysis & Visualisation

---

## Contents

- `dev/intermediate_product/main.py`: Main preliminary code for end product of this course. To be moved to src later in time.
- `dev/toolkit/main.py`: End product of course 'Python'. Part are and will be re-used in the intermediate and end product of this course.
- `src` and `img` (folders) mimic the course lectures:
      - 1. preprocessoor
      - 2. categories
      - 3. time
      - 4. distribution
- `hello.py`: Simple module for testing imports (created by {UV})
- `README.md`: Project overview and author info

---

## Setup

At this moment src/1-preprocessing/main.py runs a preprossor that converts whatapp-messages to csv & parq format.
The input file is specified in the file "config.toml" as inputpath. Example inputpath = "whatsapp-20250916-202952.csv"

Code to generate current best graphs:
- 2. Categories is in src/2b-enhanced/7-final.py. Its output is in img/yearly_bar_chart_combined.png 
- 3. Time is in src/3-time/plain/12-final.py. Its output is in img/3-golf decode by wa heartbeat.png.

The scripts in dev/toolkit are tied to a toolkit made in an earlier course (Python):
  - The main of the toolkit is renamed main_toolkit.py for now.
  - find_structures_final is a standalone script to find structures
  - find_words_final is a standalone script to find words in one or more specific columns

---

## Installation

To install dependencies and run the project, refer to the `pyproject.toml` file.  
This project uses [UV](https://github.com/astral-sh/uv) for dependency management and environment setup.

To create and sync the virtual environment:

```bash
uv venv
uv sync