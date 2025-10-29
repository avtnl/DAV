# run_enrich.py
from src.file_manager import FileManager
from src.data_editor import DataEditor
from pathlib import Path
import pandas as pd

fm = FileManager()
de = DataEditor()

processed_dir = Path("data") / "processed"
enriched_csv = fm.enrich_all_groups(de, processed_dir)

if enriched_csv and Path(enriched_csv).exists():
    print(f"\nEnrichment complete: {enriched_csv}")

    # ------------------------------------------------------------------
    # POST-ENRICHMENT: Add 'whatsapp_group_temp' (AvT isolated)
    # ------------------------------------------------------------------
    df = pd.read_csv(enriched_csv)

    # Ensure required columns exist
    if 'author' not in df.columns or 'whatsapp_group' not in df.columns:
        raise KeyError("Enriched file must contain 'author' and 'whatsapp_group' columns.")

    # Create whatsapp_group_temp: AvT becomes its own group
    df['whatsapp_group_temp'] = df['whatsapp_group'].where(
        df['author'] != "Anthony van Tilburg",
        "AvT"
    )

    # Overwrite enriched file with new column
    output_path = Path(enriched_csv)
    df.to_csv(output_path, index=False)
    print(f"Added 'whatsapp_group_temp' column → {output_path.name}")

    print(f"\nREADY! Use this file in style_analyzer:\n{output_path}")
else:
    print("Enrichment failed or no output generated.")