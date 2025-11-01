# run_enrich.py
from pathlib import Path

from src.data_editor import DataEditor
from src.file_manager import FileManager

fm = FileManager()
de = DataEditor()

processed_dir = Path("data") / "processed"
enriched_csv = fm.enrich_all_groups(de, processed_dir)

if enriched_csv and Path(enriched_csv).exists():
    print(f"\nREADY! Enriched file (with whatsapp_group_temp):\n{enriched_csv}")
else:
    print("Enrichment failed.")
