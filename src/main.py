# src/main.py
from src.scripts.pipeline import Pipeline  # ← FIXED

# ------------------------------------------------------------------
# CONFIGURE YOUR PIPELINE HERE
# ------------------------------------------------------------------
# Order matters! Script0 (preprocessing) is run automatically first.
# Only include scripts you want to execute.
SCRIPTS = [7,1,2,3,4,5,10,11]

# ------------------------------------------------------------------
if __name__ == "__main__":
    Pipeline.run(scripts=SCRIPTS)