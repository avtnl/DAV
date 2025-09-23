## Author

- Name: Anthony van Tilburg 
- Student Number: 1905780
- Institution: Hogeschool Utrecht (Netherlands) 
- Course: Master of Informatics | Applied Data Science | Data Analysis & Visualisation

---

## Contents

- `main.py`: Main preliminary script for testing some basic packages
- `hello.py`: Simple module for testing imports (created by {UV})
- `README.md`: Project overview and author info

---

## Setup

At this moment main.py runs a preprossor that converts whatapp-messages to csv & parq format.
The input file is specified in the file "config.toml" as inputpath. Example inputpath = "whatsapp-20250916-202952.csv"

The other scripts are tied to a toolkit made in an earlier course:
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