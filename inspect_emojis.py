# inspect.py
from src.file_manager import FileManager

fm = FileManager()
df = fm.get_latest_preprocessed_df()

if df is not None:
    print("First 5 rows of 'list_of_all_emojis':")
    print(df['list_of_all_emojis'].head())
    print("\nType of first entry:")
    print(type(df['list_of_all_emojis'].iloc[0]))
else:
    print("No data loaded.")